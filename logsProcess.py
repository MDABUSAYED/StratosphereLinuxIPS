# This file takes care of creating the log files with information
# Every X amount of time it goes to the database and reports

import multiprocessing
import sys
from datetime import datetime
from datetime import timedelta
import os
import threading
import time
from slips.core.database import __database__
import configparser
import pprint

def timing(f):
    """ Function to measure the time another function takes. It should be used as decorator: @timing"""
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('function took {:.3f} ms'.format((time2-time1)*1000.0))
        return ret
    return wrap

# Logs output Process
class LogsProcess(multiprocessing.Process):
    """ A class to output data in logs files """
    def __init__(self, inputqueue, outputqueue, verbose, debug, config):
        multiprocessing.Process.__init__(self)
        self.verbose = verbose
        self.debug = debug
        self.config = config
        # From the config, read the timeout to read logs. Now defaults to 5 seconds
        self.inputqueue = inputqueue
        self.outputqueue = outputqueue
        # Read the configuration
        self.read_configuration()
        self.fieldseparator = __database__.getFieldSeparator()
        # For some weird reason the database loses its outputqueue and we have to re set it here.......
        __database__.setOutputQueue(self.outputqueue)

    def read_configuration(self):
        """ Read the configuration file for what we need """
        # Get the time of log report
        try:
            self.report_time = int(self.config.get('parameters', 'log_report_time'))
        except (configparser.NoOptionError, configparser.NoSectionError, NameError):
            # There is a conf, but there is no option, or no section or no configuration file specified
            self.report_time = 5
        self.outputqueue.put('01|logs|Logs Process configured to report every: {} seconds'.format(self.report_time))

    def run(self):
        try:
            # Create our main output folder. The current datetime with microseconds
            # TODO. Do not create the folder if there is no data? (not sure how to)
            self.mainfoldername = datetime.now().strftime('%Y-%m-%d--%H:%M:%S')
            if not os.path.exists(self.mainfoldername):
                    os.makedirs(self.mainfoldername)
            # go into this folder
            os.chdir(self.mainfoldername)

            # Process the data with different strategies
            # Strategy 1: Every X amount of time
            # Create a timer to process the data every X seconds
            timer = TimerThread(self.report_time, self.process_global_data)
            timer.start()

            while True:
                if not self.inputqueue.empty():
                    line = self.inputqueue.get()
                    if 'stop' != line:
                        # we are not processing input from the queue yet
                        # without this line the complete output thread does not work!!
                        # WTF???????
                        print(line)
                        pass
                    else:
                        # Here we should still print the lines coming in the input for a while after receiving a 'stop'. We don't know how to do it.
                        self.outputqueue.put('stop')
                        return True
                elif self.inputqueue.empty():
                    # Nothing to do here either now
                    pass
            # Stop the timer
            timer.shutdown()

        except KeyboardInterrupt:
            # Stop the timer
            timer.shutdown()
            return True
        except Exception as inst:
            # Stop the timer
            timer.shutdown()
            self.outputqueue.put('01|logs|\t[Logs] Error with LogsProcess')
            self.outputqueue.put('01|logs|\t[Logs] {}'.format(type(inst)))
            self.outputqueue.put('01|logs|\t[Logs] {}'.format(inst))
            sys.exit(1)

    def createProfileFolder(self, profileid):
        """
        Receive a profile id, create a folder if its not there. Create the log files.
        """
        # Ask the field separator to the db
        profilefolder = profileid.split(self.fieldseparator)[1]
        if not os.path.exists(profilefolder):
            os.makedirs(profilefolder)
            # If we create the folder, add once there the profileid. We have to do this here if we want to do it once.
            self.addDataToFile(profilefolder + '/' + 'ProfileData.txt', 'Profileid : ' + profileid)
        return profilefolder

    def addDataToFile(self, filename, data, file_mode='w+', data_type='txt', data_mode='text'):
        """
        Receive data and append it in the general log of this profile
        If the filename was not opened yet, then open it, write the data and return the file object.
        Do not close the file
        In data_mode = 'text', we add a \n at the end
        In data_mode = 'raw', we do not add a \n at the end
        """
        if data_type == 'json':
            # Implement some fancy print from json data
            data = data
        if data_mode == 'text':
            data = data + '\n'
        try:
            filename.write(data)
            return filename
        except (NameError, AttributeError) as e:
            # The file was not opened
            fileobj = open(filename, file_mode)
            fileobj.write(data)
            # For some reason the files are closed and flushed correclty.
            return fileobj
        except KeyboardInterrupt:
            return True

    def process_global_data(self):
        """ 
        This is the main function called by the timer process
        Read the global data and output it on logs 
        """
        try:
            #1. Get the list of profiles so far
            #temp_profs = __database__.getProfiles()
            #if not temp_profs:
            #    return True
            #profiles = list(temp_profs)

            # How many profiles we have?
            profilesLen = str(__database__.getProfilesLen())
            # Get the list of all the modifed TW for all the profiles
            TWforProfile = __database__.getModifiedTW()
            amount_of_modified = len(TWforProfile)
            self.outputqueue.put('20|logs|[Logs] Number of Profiles in DB: {}. Modified TWs: {}. ({})'.format(profilesLen, amount_of_modified , datetime.now().strftime('%Y-%m-%d--%H:%M:%S')))
            for profileTW in TWforProfile:
                # Get the profileid and twid
                #profileTW = profileTW.decode('utf-8')
                profileid = profileTW.split(self.fieldseparator)[0] + self.fieldseparator + profileTW.split(self.fieldseparator)[1]
                twid = profileTW.split(self.fieldseparator)[2]
                # Get the time of this TW. For the file name
                twtime = __database__.getTimeTW(profileid, twid)
                twtime = time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(twtime))
                self.outputqueue.put('02|logs|\t[Logs] Storing Profile: {}. TW {}. Time: {}'.format(profileid, twid, twtime))
                #self.outputqueue.put('30|logs|\t[Logs] Profile: {} has {} timewindows'.format(profileid, twLen))
                # Create the folder for this profile if it doesn't exist
                profilefolder = self.createProfileFolder(profileid)
                twlog = twtime + '.' + twid
                # Add data into profile log file
                # First Erase its file and save the data again
                self.addDataToFile(profilefolder + '/' + twlog, '', file_mode='w+', data_mode = 'raw')
                #
                # Save in the log file all parts of the profile
                # 1. DstIPs
                dstips = __database__.getDstIPsfromProfileTW(profileid, twid)
                if dstips:
                    # Add dstips to log file
                    self.addDataToFile(profilefolder + '/' + twlog, 'DstIP:\n' + dstips, file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs] DstIP: ' + dstips)
                # 2. SrcIPs
                srcips = __database__.getSrcIPsfromProfileTW(profileid, twid)
                if srcips:
                    # Add srcips
                    self.addDataToFile(profilefolder + '/' + twlog, 'SrcIP:\n'+ srcips, file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs] SrcIP: ' + srcips)
                # 3. Tuples
                tuples = __database__.getOutTuplesfromProfileTW(profileid, twid)
                if tuples:
                    # Add tuples
                    self.addDataToFile(profilefolder + '/' + twlog, 'OutTuples:\n'+ tuples, file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs] Tuples: ' + tuples)
                # 4. Detections to block
                blocking = __database__.getBlockingRequest(profileid, twid)
                if blocking:
                    self.addDataToFile(profilefolder + '/' + twlog, 'Blocking Request:\n' + str(blocking), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs] Blocking Request: ' + str(blocking))
                # 5. Info of dstport as client, tcp, established
                dstportdata = __database__.getDstPortClientTCPEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortTCPEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 6. Info of dstport as client, udp, established
                dstportdata = __database__.getDstPortClientUDPEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortUDPEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 7. Info of dstport as client, tcp, notestablished
                dstportdata = __database__.getDstPortClientTCPNotEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortTCPNotEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 8. Info of dstport as client, udp, notestablished
                dstportdata = __database__.getDstPortClientUDPNotEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortUDPNotEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 9. Info of dstport as client, icmp, established
                dstportdata = __database__.getDstPortClientICMPEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortICMPEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 10. Info of dstport as client, icmp, notestablished
                dstportdata = __database__.getDstPortClientICMPNotEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortICMPNotEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 11. Info of dstport as client, ipv6-icmp, established
                dstportdata = __database__.getDstPortClientIPV6ICMPEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortIPV6ICMPEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))
                # 12. Info of dstport as client, ipv6icmp, notestablished
                dstportdata = __database__.getDstPortClientIPV6ICMPNotEstablishedFromProfileTW(profileid, twid)
                if dstportdata:
                    self.addDataToFile(profilefolder + '/' + twlog, 'ClientDstPortIPV6ICMPNotEstablished:\n' + str(dstportdata), file_mode='a+', data_type='json')
                    self.outputqueue.put('03|logs|\t\t[Logs]: DstPortData: {}'.format(dstportdata))


                # Mark it as not modified anymore
                __database__.markProfileTWAsNotModified(profileid, twid)
        except KeyboardInterrupt:
            return True
        except Exception as inst:
            self.outputqueue.put('01|logs|\t[Logs] Error in process_global_data in LogsProcess')
            self.outputqueue.put('01|logs|\t[Logs] {}'.format(type(inst)))
            self.outputqueue.put('01|logs|\t[Logs] {}'.format(inst))
            sys.exit(1)


class TimerThread(threading.Thread):
    """Thread that executes a task every N seconds. Only to run the process_global_data."""
    
    def __init__(self, interval, function):
        threading.Thread.__init__(self)
        self._finished = threading.Event()
        self._interval = interval
        self.function = function 

    def shutdown(self):
        """Stop this thread"""
        self._finished.set()
    
    def run(self):
        try:
            while 1:
                if self._finished.isSet(): return
                self.task()
                
                # sleep for interval or until shutdown
                self._finished.wait(self._interval)
        except KeyboardInterrupt:
            return True
    
    def task(self):
        self.function()
