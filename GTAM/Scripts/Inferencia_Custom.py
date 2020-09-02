# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 01:21:45 2020

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

test_df = pd.read_excel('Test/GRTs-0108-1208.xlsx')
test_df = test_df[test_df['VICEPRESIDENCIA']== 'GERENCIAS VPEO']
test_df= test_df[pd.notnull(test_df['EVENT_DESC'])]
test_df= test_df[pd.notnull(test_df['INC_DESC'])]
test_df['EVENT_DESC']= test_df['EVENT_DESC'].astype(str)
test_df['INC_DESC']= test_df['INC_DESC'].astype(str)
test_df['texto'] = test_df['EVENT_DESC']+test_df['INC_DESC']
test_df= test_df[pd.notnull(test_df['texto'])]
test_df.reset_index(inplace = True, drop = True)

test_comments = list(test_df.texto.values)[0:100]

max_length = 100#tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True) 
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True) 
model = BertModel.from_pretrained("bert-base-multilingual-uncased")

# Encoding input data
test_encodings = tokenizer.batch_encode_plus(test_comments,max_length=max_length,pad_to_max_length=True)
test_input_ids = test_encodings['input_ids']
test_token_type_ids = test_encodings['token_type_ids']
test_attention_masks = test_encodings['attention_mask']


# Tensores
test_inputs = torch.tensor(test_input_ids)
test_masks = torch.tensor(test_attention_masks)
test_token_types = torch.tensor(test_token_type_ids)

outputs = model(test_inputs, token_type_ids=None, attention_mask= test_masks)
data_process = outputs[1].detach().numpy()
data_out =pd.DataFrame(data_process)
data_out.to_excel("Inferencias/test_grt_fix.xlsx", index = False)

auxInc = pd.read_excel("auxgrt.xlsx")

auxInc["texto"]= test_comments

auxInc.to_excel("auxgrt.xlsx",index = False)