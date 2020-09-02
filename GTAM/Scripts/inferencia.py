# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:35:32 2020

author: AMS
"""


# Imports
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertModel



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)



test_df = pd.read_excel('Test/GRTs-0108-1208.xlsx')
test_df = test_df[test_df['VICEPRESIDENCIA']== 'GERENCIAS VPEO']
test_df= test_df[pd.notnull(test_df['EVENT_DESC'])]
test_df= test_df[pd.notnull(test_df['INC_DESC'])]
test_df['EVENT_DESC']= test_df['EVENT_DESC'].astype(str)
test_df['INC_DESC']= test_df['INC_DESC'].astype(str)
test_df['texto'] = test_df['EVENT_DESC'] + test_df['INC_DESC']+test_df['DESVC6_DESCRIPCION'].fillna('')+test_df['DESVC1_DESCRIPCION'].fillna('')+test_df['DESVC2_DESCRIPCION'].fillna('')+test_df['DESVC3_DESCRIPCION'].fillna('')+test_df['DESVC4_DESCRIPCION'].fillna('')+test_df['DESVC5_DESCRIPCION'].fillna('')
test_df= test_df[pd.notnull(test_df['texto'])]
test_df.reset_index(inplace = True, drop = True)


print('Null values: ', test_df.isnull().values.any())
test_df.head()

test_comments = list(test_df.texto.values)

max_length = 100
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


batch_size = 8

# Create test dataloader
test_data = TensorDataset(test_inputs, test_masks, test_token_types)# test_labels,
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Test

model.eval()

#track variables
logit_preds,pred_labels,tokenized_texts = [],[],[]

# Predict
for i, batch in enumerate(test_dataloader):
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_token_types = batch
  with torch.no_grad():
    # Forward pass
    outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)
    b_logit_pred = b_logit_pred.detach().cpu().numpy()
    pred_label = pred_label.to('cpu').numpy()
    #b_labels = b_labels.to('cpu').numpy()

  tokenized_texts.append(b_input_ids)
  logit_preds.append(b_logit_pred)
  pred_labels.append(pred_label)

# Flatten outputs
tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]

# Definir el nombre de las columnas de salida
label_cols = ['Defciente controles  plan Covid 19',
 'Falta internalización del CGR',
 'Ausencia o deficiencia plan específico de bloqueo',
 'VATS deficiente',
 'Ejecución deficiente de la actividad ',
 'Instructivo  deficiente',
 'Ausencia de planificación',
 'Aprendizaje de evento anterior no internalizado',
 'Controles  no aplicados correctamente',
 'Trabajos en caliente',
 'Orden y aseo deficiente en área de trabajo',
 'Residuos industriales  peligrosos -domésticos',
 'Falta de stock de respuestos',
 'Carga, transporte y descarga',
 'Maniobras de izaje',
 'Retraso en inicio de los trabajos',
 'Energía de orígen y potencial no identificada ',
 'Entorno de trabajo no controlado ',
 'Conducción']


pred_bools = [pl>0.6 for pl in pred_labels]
test_df_bool = pd.DataFrame(data = pred_bools, columns = label_cols)
test_df_prob = pd.DataFrame(data = pred_labels, columns = label_cols)
test_df_int = test_df_bool.astype(int)

#recuperar informacion de la grt
test_df_concat = pd.concat([test_df, test_df_int], axis =1)
test_df_concat_prob = pd.concat([test_df, test_df_prob], axis = 1)