#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:10:07 2024

@author: msayed
"""

import pandas as pd
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
import time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

startTime = time.time() 
# URL of the raw CSV file
data_url = 'mixed_domain.csv'

try:
    
    # Read the CSV file directly from the URL
    df = pd.read_csv(data_url)
    print('Data Loaded successfully from github repository...')
except Exception:
    print('Error in data loading. Please run the program again..')

# Display the first few rows of the DataFrame
print('DataFrame shape :', df.shape)
print('First few rows of data : \n', df.head())

# Create a balanced dataset
balanced_df = df.groupby('Type').sample(n=3000, random_state=1).reset_index(drop=True)  # random_state for reproducibility

# Display the first few rows of the balanced DataFrame
print('Balanced DataFrame shape :', balanced_df.shape)
print('First few rows of data : \n', balanced_df.head())

print('Count of Type from Balanced Dataset (DGA and Normal) :\n',balanced_df['Type'].value_counts())

# Define and communicate llama3 LLM model

def prompt_answer(msg):
    messages = [{'role':"user", "content":msg},]


    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        #eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

    
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)




'''
prompt = 'Is ' + 'www.msftncsi.com ' + 'a DGA domain? Please give me yes or no answer?'
response  = llama3(prompt)

print(response)
'''

# Ask llama3 about domain whether DGA or not
balanced_df['LLM_output'] = balanced_df['Domain'].apply(lambda x: prompt_answer(" Is " + x + "a DGA domain? Please give me yes or no answer?"))


# Preprocess LLM output to DGA and Normal types

balanced_df['LLM_output'] = balanced_df['LLM_output'].str.replace('[*.]', '', regex=True)
balanced_df['LLM_output'] = balanced_df['LLM_output'].str.lower()
balanced_df['LLM_output'] = balanced_df['LLM_output'].str.replace('yes', 'DGA')
balanced_df['LLM_output'] = balanced_df['LLM_output'].str.replace('no', 'Normal')


print('Count of Type from LLM output (DGA and Normal) :\n',balanced_df['LLM_output'].value_counts())

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(balanced_df['Type'], balanced_df['LLM_output']).ravel()

# Accuracy
accuracy = (tp + tn) / float(tp + tn + fp + fn)

# Precision (Positive Predictive Value)
precision = tp / float(tp + fp)

# Recall (Sensitivity or True Positive Rate)
recall = tp / float(tp + fn)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(balanced_df['Type'], balanced_df['LLM_output'])

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

print('Pretrained llam3 performance on different matrics\n')
print("Sensitivity (Recall):", recall)
print("Specificity:", specificity)
print("Precision (Positive Predictive Value):", precision)
print("Negative Predictive Value:", npv)
print("False Positive Rate:", fpr)
print("False Discovery Rate:", fdr)
print("False Negative Rate:", fnr)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)

print("Total time taken in seconds = ", str(time.time() - startTime))