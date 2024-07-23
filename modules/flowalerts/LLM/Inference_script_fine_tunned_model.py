#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:41:10 2024

@author: msayed
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import time
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    pipeline,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_compute_dtype=torch.bfloat16  # Set compute dtype to bfloat16
)


startTime = time.time() 

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# URL of the raw CSV file
data_url = 'mixed_domain_test_2k.csv'

try:
    
    # Read the CSV file directly from the URL
    df = pd.read_csv(data_url)
    print('Data Loaded successfully ...')
except Exception:
    print('Error in data loading. Please run the program again..')

# Display the first few rows of the DataFrame
print('DataFrame shape :', df.shape)
print('First few rows of data : \n', df.head())


# # create new dataset
dataset = DatasetDict({'test':Dataset.from_dict({'label':df['Type'].tolist(),'text':df['Domain'].tolist()})})



print('Dataset dict : ', dataset)

# define label maps
id2label = {0: 'No', 1: 'Yes'}
label2id = {'No': 0, 'Yes': 1}

# Load finetunned classification model locally

#model_name = './Roberta-Base-TrainedModel'
model_id = int(input("Enter number for different fine-tunned model \n 1 for lora Roberta-Base \n\
 2 for lora  Distilbert-Base\n 3 for qlora - Meta-Llama-3-8B \n 4 for qlora - Zephyr-7B-Beta  \n"))

if model_id == 1:
    model_name = './Roberta-Base-TrainedModel'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
elif model_id == 2:
    model_name = './Distilbert-Base-TrainedModel'
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
elif model_id == 3:
    model_name = './Meta-Llama-3-8B-TrainedModel'
    model = AutoModelForSequenceClassification.from_pretrained(model_name,quantization_config=bnb_config,\
                                    num_labels=2, id2label=id2label,label2id=label2id, device_map="auto")
    
elif model_id == 4:
    model_name = './Zephyr-7B-Beta-TrainedModel'
    model = AutoModelForSequenceClassification.from_pretrained(model_name,quantization_config=bnb_config,\
                                    num_labels=2, id2label=id2label,label2id=label2id, device_map="auto")


print('Model Description: ', model)

#print('Model trainable parameteres : ', model.print_trainable_parameters())

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

model.config.pad_token_id = model.config.eos_token_id

clf = pipeline("text-classification", model=model, tokenizer=tokenizer)


# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
print("Tokenized Test Dataset : ", tokenized_dataset)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

'''
# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}
'''
# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # Accuracy
    acc = (tp + tn) / float(tp + tn + fp + fn)

    # Precision (Positive Predictive Value)
    precision = tp / float(tp + fp)

    # Recall (Sensitivity or True Positive Rate)
    recall = tp / float(tp + fn)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(labels, preds)

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp)

    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0

    # False Positive Rate
    fpr = fp / (fp + tn)

    # False Discovery Rate
    fdr = fp / (fp + tp) if (fp + tp) != 0 else 0

    # False Negative Rate
    fnr = fn / (fn + tp)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews_corrcoef": mcc,
        "specificity": specificity,
        "npv": npv,
        "fpr": fpr,
        "fdr": fdr,
        "fnr": fnr
    }

# Define training arguments

training_args = TrainingArguments(
    output_dir= model_name + "-Inference",
    per_device_eval_batch_size=8,
    do_eval=True,
    logging_steps=1,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print(results)

print("Total time taken in seconds = ", str(time.time() - startTime))

