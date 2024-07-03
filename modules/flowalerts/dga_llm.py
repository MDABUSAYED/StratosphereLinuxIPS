import pandas as pd

#import transformers
import torch

import requests
import json
import time

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

startTime = time.time() 
# URL of the raw CSV file
data_url = 'https://raw.githubusercontent.com/hmaccelerate/DGA_Detection/master/data/mixed_domain.csv'

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
model_url = "http://localhost:11434/api/chat"

def llama3(prompt):
    data = {
        "model": "llama3",
        "messages": [
            {
              "role": "user",
              "content": prompt
            }
        ],
        "stream": False
    }
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.post(model_url, headers=headers, json=data)
    
    return(response.json()['message']['content'])

'''
prompt = 'Is ' + 'www.msftncsi.com ' + 'a DGA domain? Please give me yes or no answer?'
response  = llama3(prompt)

print(response)
'''

# Ask llama3 about domain whether DGA or not
balanced_df['LLM_output'] = balanced_df['Domain'].apply(lambda x: llama3(" Is " + x + "a DGA domain? Please give me yes or no answer?"))


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