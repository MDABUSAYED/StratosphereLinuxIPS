#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 21:01:23 2024

@author: msayed
"""

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model


import pandas as pd
from sklearn.model_selection import train_test_split
import time
from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef



startTime = time.time() 
'''
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
'''

# URL of the raw CSV file
data_url = 'mixed_domain.csv'

try:
    
    # Read the CSV file directly from the URL
    df = pd.read_csv(data_url)
    print('Data Loaded successfully ...')
except Exception:
    print('Error in data loading. Please run the program again..')

# Display the first few rows of the DataFrame
print('DataFrame shape :', df.shape)
print('First few rows of data : \n', df.head())

num_data = int(input("Enter a number of data you want to consider for DGA and Normal type domain(individually) : "))

balanced_df = df.groupby('Type').sample(n=num_data, random_state=1).reset_index(drop=True)

#balanced_df['Type'] = balanced_df['Type'].str.replace('DGA', 1)
#balanced_df['Type'] = balanced_df['Type'].str.replace('Normal', 0)

balanced_df['Type'] = balanced_df['Type'].replace({'DGA': 1, 'Normal': 0})

# Display the first few rows of the balanced DataFrame
print('Balanced DataFrame shape :', balanced_df.shape)
print('First few rows of data : \n', balanced_df.head())


# Shuffle the DataFrame
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print('Now spliting dataset as training (60%), validation(20%), and test(20%) and creating datasetdict')
# Separate the data based on type
df_type1 = balanced_df[balanced_df['Type'] == 1]
df_type2 = balanced_df[balanced_df['Type'] == 0]

# Split type1 data
type1_train, type1_temp = train_test_split(df_type1, test_size=0.4, random_state=42)
type1_val, type1_test = train_test_split(type1_temp, test_size=0.5, random_state=42)

# Split type2 data
type2_train, type2_temp = train_test_split(df_type2, test_size=0.4, random_state=42)
type2_val, type2_test = train_test_split(type2_temp, test_size=0.5, random_state=42)

# Combine type1 and type2 data for each set
train_df = pd.concat([type1_train, type2_train]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = pd.concat([type1_val, type2_val]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat([type1_test, type2_test]).sample(frac=1, random_state=42).reset_index(drop=True)

# # create new dataset
dataset = DatasetDict({'train':Dataset.from_dict({'label':train_df['Type'].tolist(),'text':train_df['Domain'].tolist()}),\
                       'validation':Dataset.from_dict({'label':val_df['Type'].tolist(),'text':val_df['Domain'].tolist()}), \
                       'test':Dataset.from_dict({'label':test_df['Type'].tolist(),'text':test_df['Domain'].tolist()})})



print('Dataset dict : ', dataset)



model_name = "HuggingFaceH4/zephyr-7b-beta"
# "meta-llama/Meta-Llama-3-8B" # "unsloth/llama-3-8b-bnb-4bit" #"HuggingFaceH4/zephyr-7b-beta" # "meta-llama/Meta-Llama-3-8B" #'distilbert-base-uncased'
# model_checkpoint = 'roberta-base' # you can alternatively use roberta-base but this model is bigger thus training will take longer


quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, # enable 4-bit quantization
    bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
    bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
    bnb_4bit_compute_dtype = torch.bfloat16, # optimized fp format for ML)
)

lora_config = LoraConfig(
    r = 16, # the dimension of the low-rank matrices
    lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.01, # dropout probability of the LoRA layers
    bias = 'none', # wether to train bias weights, set to 'none' for attention layers
    task_type = 'SEQ_CLS',
)
     

# define label maps
id2label = {0: 'No', 1: 'Yes'}
label2id = {'No': 0, 'Yes': 1}


# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2, id2label=id2label, label2id=label2id)


#model.config.pad_token_id = model.config.eos_token_id

print('Model : ', model)


model = prepare_model_for_kbit_training(model)

print('prepare_model_for_kbit_training Model :\n ', model)

model = get_peft_model(model, lora_config)

print('get_peft_model Model :\n ', model)



tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

'''
# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
'''

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1


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
print("Tokenized Dataset : ", tokenized_dataset)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

'''
peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=8,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"])
print('Peft Config : ', peft_config)

model = get_peft_model(model, peft_config)
print('Peft model description : ', model.print_trainable_parameters())
'''
print('Peft model description : ', model.print_trainable_parameters())

# hyperparameters
lr = 1e-4
batch_size = 8
num_epochs = 4

# define training arguments
training_args = TrainingArguments(
    output_dir= model_name + "-qlora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
    
)

'''
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
'''

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
#model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

# renable warnings
#model.config.use_cache = True

# Evaluate model on the test dataset
predictions = trainer.predict(tokenized_dataset["test"])

# Compute metrics
preds = np.argmax(predictions.predictions, axis=-1)

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(tokenized_dataset["test"]['label'], preds).ravel()

# Accuracy
accurac = (tp + tn) / float(tp + tn + fp + fn)

# Precision (Positive Predictive Value)
precision = tp / float(tp + fp)

# Recall (Sensitivity or True Positive Rate)
recall = tp / float(tp + fn)

# F1 Score
f1 = 2 * (precision * recall) / (precision + recall)

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(tokenized_dataset["test"]['label'], preds)

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

print('Finetuned Q-lora on Zephyr model performance on different matrics\n')
print("Sensitivity (Recall):", recall)
print("Specificity:", specificity)
print("Precision (Positive Predictive Value):", precision)
print("Negative Predictive Value:", npv)
print("False Positive Rate:", fpr)
print("False Discovery Rate:", fdr)
print("False Negative Rate:", fnr)
print("Accuracy:", accurac)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)

print("Total time taken in seconds = ", str(time.time() - startTime))


    
