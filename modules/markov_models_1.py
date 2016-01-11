# This file is part of the Stratosphere Linux IPS
# See the file 'LICENSE' for copying permission.

from colors import *
import cPickle
import math

class Model():
    def __init__(self, id):
        self.id = id
        self.init_vector = False
        self.matrix = False
        self.self_probability = -1
        self.label = -1

    def compute_probability(self, state):
        """ Given a chain of letters, return the probability that it was generated by this MC """
        i = 0
        probability = 0
        ignored = 0
        # Get the initial probability of this letter in the IV.
        try:
            init_letter_prob = math.log(self.init_vector[state[i]])
        except ValueError:
            # There is not enough data to even create a matrix
            init_letter_prob = 0
        except IndexError:
            # The first letter is not in the matrix, so penalty...
            init_letter_prob = -4.6
        # We should have more than 2 states at least
        while i < len(state) and len(state) > 1:
            try:
                vector = state[i] + state[i+1]
                growing_v = state[0:i+2]
                # The transitions that include the # char will be automatically excluded
                temp_prob = self.matrix.walk_probability(vector)
                i += 1
                if temp_prob != float('-inf'):
                    probability = probability + temp_prob # logs should be summed up
                    #print_info('\tTransition [{}:{}]: {} -> Prob:{:.10f}. CumProb: {}'.format(i-1, i,vector, temp_prob, probability))
                else:
                    # Here is our trick. If two letters are not in the matrix... assign a penalty probability
                    # The temp_prob is the penalty we assign if we can't find the transition
                    temp_prob = -4.6 # Which is approx 0.01 probability
                    #temp_prob = -20 # Which is approx 0.01 probability
                    #temp_prob = -40 # Which is approx 0.01 probability
                    #temp_prob = -80 # Which is approx 0.01 probability
                    probability = probability + temp_prob # logs should be +
                    if '#' not in vector:
                        ignored += 1
                    continue
            except IndexError:
                # We are out of letters
                break
        #if ignored:
            #print_warning('Ignored transitions: {}'.format(ignored))
            #ignored = 0
        return probability


    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def get_id(self):
        return self.id

    def set_init_vector(self, vector):
        self.init_vector = vector

    def get_init_vector(self):
        return self.init_vector

    def set_matrix(self, matrix):
        self.matrix = matrix

    def get_matrix(self):
        return self.matrix

    def set_self_probability(self, prob):
        self.self_probability = prob

    def get_self_probability(self):
        return self.self_probability

    def set_label(self, label):
        self.label = label
        protocol = label.split('-')[2]
        self.set_protocol(protocol)
        # Set the responce that should be given if matched
        if 'normal' in label.lower():
            self.matched = False
        else:
            self.matched = True

    def get_label(self):
        return self.label

    def set_protocol(self, protocol):
        self.protocol = protocol

    def get_protocol(self):
        return self.protocol

    def set_threshold(self, threshold):
        self.threshold = threshold

    def get_threshold(self):
        return self.threshold


class MarkovModelsDetection():
    """
    Class that do all the detection using markov models
    """
    def __init__(self):
        self.models = []

    def is_periodic(self,state):
        basic_patterns = ['a,a,a,','b,b,b,', 'c,c,c,', 'd,d,d,', 'e,e,e,', 'f,f,f,', 'g,g,g,', 'h,h,h,', 'i,i,i,', 'a+a+a+', 'b+b+b+', 'c+c+c+', 'd+d+d+', 'e+e+e+', 'f+f+f+', 'g+g+g+', 'h+h+h+', 'i+i+i+', 'a*a*a*', 'b*b*b*', 'c*c*c*', 'd*d*d*', 'e*e*e*', 'f*f*f*', 'g*g*g*', 'h*h*h*', 'i*i*i*', 'A,A,A,','B,B,B,', 'C,C,C,', 'D,D,D,', 'E,E,E,', 'F,F,F,', 'G,G,G,', 'H,H,H,', 'I,I,I,', 'A+A+A+', 'B+B+B+', 'C+C+C+', 'D+D+D+', 'E+E+E+', 'F+F+F+', 'G+G+G+', 'H+H+H+', 'I+I+I+', 'A*A*A*', 'B*B*B*', 'C*C*C*', 'D*D*D*', 'E*E*E*', 'F*F*F*', 'G*G*G*', 'H*H*H*', 'I*I*I*']
        for pattern in basic_patterns:
            if pattern in state:
                return True

    def set_model_to_detect(self, file):
        """
        Receives a file and extracts the model in it
        """
        input = open(file, 'r')
        try:
            id = self.models[-1].get_id() + 1
        except (KeyError, IndexError):
            id = 1
        model = Model(id)
        model.set_init_vector(cPickle.load(input))
        model.set_matrix(cPickle.load(input))
        model.set_state(cPickle.load(input))
        model.set_self_probability(cPickle.load(input))
        model.set_label(cPickle.load(input))
        model.set_threshold(cPickle.load(input))
        self.models.append(model)
        print 'Adding model {} to the list.'.format(model.get_label())
        input.close()

    def detect(self, tuple, verbose):
        """
        Main detect function
        """
        try:
            # Clear the temp best model
            best_model_so_far = False
            best_distance_so_far = float('inf')
            best_model_matching_len = -1
            # Set the verbose
            self.verbose = verbose
            # Only detect states with more than 3 letters
            if len(tuple.get_state()[tuple.get_max_state_len():]) < 4:
                if self.verbose > 3:
                    print '\t-> State too small'
                return (False, False)
            # Use the current models for detection
            for model in self.models:
                # Only detect if protocol matches
                if model.get_protocol().lower() != tuple.get_protocol().lower():
                    # Go get the next
                    continue
                # Letters of the trained model. Get from the last detected letter to the end. NO CUT HERE. We dont cut the training letters, because if we do, we have to cut ALL of them,
                # including the matching and the not matching ones.
                train_sequence = model.get_state()[0:len(tuple.get_state())]
                # We dont recreate the matrix because the trained is never cutted.
                # Get the new original prob so far...
                training_original_prob = model.compute_probability(train_sequence)
                # Now obtain the probability for testing. The prob is computed by using the API on the train model, which knows its own matrix
                test_prob = model.compute_probability(tuple.get_state()[tuple.get_max_state_len():])
                # Get the distance
                prob_distance = -1
                if training_original_prob != -1 and test_prob != -1 and training_original_prob <= test_prob:
                    try:
                        prob_distance = training_original_prob / test_prob
                    except ZeroDivisionError:
                        prob_distance = -1
                elif training_original_prob != -1 and test_prob != -1 and training_original_prob > test_prob:
                    try:
                        prob_distance = test_prob / training_original_prob
                    except ZeroDivisionError:
                        prob_distance = -1
                if self.verbose > 2:
                    print '\t\tTrained Model: {}. Label: {}. Threshold: {}, State: {}'.format(model.get_id(), model.get_label(), model.get_threshold(), train_sequence)
                    print '\t\t\tTest Model: {}. State: {}'.format(tuple.get_id(), tuple.get_state()[tuple.get_max_state_len():])
                    print '\t\t\tTrain prob: {}'.format(training_original_prob)
                    print '\t\t\tTest prob: {}'.format(test_prob)
                    print '\t\t\tDistance: {}'.format(prob_distance)
                    if self.verbose > 4:
                        print '\t\t\tTrained Matrix:'
                        matrix = model.get_matrix()
                        for i in matrix:
                            print '\t\t\t\t{}:{}'.format(i, matrix[i])
                # If we matched and we are the best so far
                if prob_distance >= 1 and prob_distance <= model.get_threshold() and prob_distance < best_distance_so_far:
                    best_model_so_far = model
                    best_distance_so_far = prob_distance
                    best_model_matching_len = len(tuple.get_state())
                    if self.verbose > 3:
                        print '\t\t\t\tThis model is the best so far. State len: {}'.format(best_model_matching_len)
            # If we detected something
            if best_model_so_far:
                # Move the states of the tuple so next time for this tuple we don't compare from the start
                if tuple.get_max_state_len() == 0:
                    # First time matched. move only the max state value of the tuple to the place where we detected the match
                    #tuple.set_max_state_len(best_model_matching_len)
                    #if self.verbose > 3:
                    #    print 'We moved the max to: {}'.format(best_model_matching_len)
                    pass
                else:
                    # Not the first time this tuple is matched. We should move the min and max
                    #tuple.set_min_state_len(tuple.get_max_state_len())
                    #tuple.set_max_state_len(best_model_matching_len)
                    #if self.verbose > 3:
                    #    print 'We moved the min to: {} and max to: {}'.format(tuple.get_max_state_len(), best_model_matching_len)
                    pass
                # Return
                #if self.verbose > 3:
                    #print 'Returning the best model with label {} ({})'.format(best_model_so_far.get_label(), best_model_so_far.matched)
                return (best_model_so_far.matched, best_model_so_far.get_label())
            else:
                return (False, False)
        except Exception as inst:
            print 'Problem in detect()'
            print type(inst)     # the exception instance
            print inst.args      # arguments stored in .args
            print inst           # __str__ allows args to printed directly
            sys.exit(-1)



__markov_models__ = MarkovModelsDetection()
