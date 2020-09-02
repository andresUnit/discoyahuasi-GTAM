# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 02:14:55 2020

author: AMS
"""


# Imports
import pandas as pd
import torch
from transformers import BertModel , BertTokenizer


# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


# Carga de Datos

df = pd.read_excel('Base Datos/data_GTAM_fix.xlsx')
#df.fillna(0, inplace = True)

# Descriptivo Basico

print('Unique comments: ', df.texto.nunique() == df.shape[0])
print('Null values: ', df.isnull().values.any())
print('average sentence length: ', df.texto.str.split().str.len().mean())
print('stdev sentence length: ', df.texto.str.split().str.len().std())

# columnas de Label
cols = df.columns
label_cols = list(cols[-20:-1])
num_labels = len(label_cols)
print('Label columns: ', label_cols)

# Descriptivo conteos
print('Count of 1 per label: \n', df[label_cols].sum(), '\n')
print('Count of 0 per label: \n', df[label_cols].eq(0).sum())


# Data Base Line

df.dropna(thresh=1, subset=label_cols, inplace = True)
df.fillna(0, inplace = True)

# Separar la data

cota = 100
label_cols_menor = list(df[label_cols].sum()[df[label_cols].sum() < cota].index)
num_labels_menor = len(label_cols_menor)

for foco in label_cols_menor:
    data = df[df[foco]==1]
    data.to_excel(f"Base Datos/Data Balanceada/{foco}.xlsx", index = False)

max_length = 100#tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True) 
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True) 
model = BertModel.from_pretrained("bert-base-multilingual-uncased")



# Generacion de Data a Utilizar

for foco in label_cols_menor:

    df = pd.read_excel(f"Base Datos/Data Balanceada/{foco}.xlsx")
    comments = list(df.texto.values)
    encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True, truncation = True)
    input_ids = encodings['input_ids'] # frases tokenizadas y encoding
    token_type_ids = encodings['token_type_ids'] # separacion de frases de bert
    attention_masks = encodings['attention_mask'] # attention masks
    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    attention_masks = torch.tensor(attention_masks)
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
    data_process = outputs[1].detach().numpy()
    data_out =pd.DataFrame(data_process)
    data_out.to_excel(f"Base Datos/Data Balanceada Bert/{foco}.xlsx", index = False)
    
    

label_cols_mayor = list(df[label_cols].sum()[df[label_cols].sum() >= cota].index)
num_labels_menor = len(label_cols_mayor)

for foco in label_cols_mayor:
    data = df[df[foco]==1]
    data = data.sample(n=100)
    data.to_excel(f"Base Datos/Data Balanceada/{foco}.xlsx", index = False)
    
for foco in label_cols_mayor:

    foco = "Entorno de trabajo no controlado "
    df = pd.read_excel(f"Base Datos/Data Balanceada/{foco}.xlsx")
    comments = list(df.texto.values)
    encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True, truncation = True)
    input_ids = encodings['input_ids'] # frases tokenizadas y encoding
    token_type_ids = encodings['token_type_ids'] # separacion de frases de bert
    attention_masks = encodings['attention_mask'] # attention masks
    input_ids = torch.tensor(input_ids)
    token_type_ids = torch.tensor(token_type_ids)
    attention_masks = torch.tensor(attention_masks)
    outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
    data_process = outputs[1].detach().numpy()
    data_out =pd.DataFrame(data_process)
    data_out.to_excel(f"Base Datos/Data Balanceada Bert/{foco}.xlsx", index = False)