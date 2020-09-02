# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:22:21 2020

author: AMS
"""


# Imports
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertForSequenceClassification , BertTokenizer, AdamW
from tqdm import trange

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

df[label_cols].sum()[0:29].sort_values().plot.barh(grid = True, title = 'Gráfico Distribución Labels Focos Transversales', xlim = (0,330))

df[label_cols].sum()[29:55].sort_values().plot.barh(grid = True, title = 'Gráfico Distribución Labels Operaciones TAM', xlim = (0,90))

df[label_cols].sum()[56:].sort_values().plot.barh(grid = True, title = 'Gráfico Distribución Labels Infraestructura y transporte de fluido', xlim = (0,90))


# Data Base Line
cota = 20
label_cols = list(df[label_cols[0:29]].sum()[df[label_cols].sum() > cota].index)
num_labels = len(label_cols)

df.dropna(thresh=1, subset=label_cols, inplace = True)
df.fillna(0, inplace = True)

pesos = torch.tensor(df[label_cols].sum().max()/df[label_cols].sum()[label_cols].values)


#shuffle
df = df.sample(frac=1).reset_index(drop=True) 

# Generacion de los labels formato One Hot
df['one_hot_labels'] = list(df[label_cols].values)
df.head()

# Generacion de Data a Utilizar

labels = list(df.one_hot_labels.values)
comments = list(df.texto.values)

# tokenizador y encoding

max_length = 150#tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True) 
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True) 
encodings = tokenizer.batch_encode_plus(comments,max_length=max_length,pad_to_max_length=True, truncation = True)
print('tokenizer outputs: ', encodings.keys())

input_ids = encodings['input_ids'] # frases tokenizadas y encoding
token_type_ids = encodings['token_type_ids'] # separacion de frases de bert
attention_masks = encodings['attention_mask'] # attention masks

# frecuencia uno
label_counts = df.one_hot_labels.astype(str).value_counts()
print('Freq: \n', label_counts)

# encontramos el indice y retornamos
one_freq = label_counts[label_counts==1].keys()
one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
print('df label indices with only one instance: ', one_freq_idxs)

# extraemos el valor y lo guardamos
one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
one_freq_token_types = [token_type_ids.pop(i) for i in one_freq_idxs]
one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
one_freq_labels = [labels.pop(i) for i in one_freq_idxs]

# Data Split

train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = train_test_split(input_ids, labels, token_type_ids,attention_masks,
random_state=2020, test_size=0.20, stratify = labels)

# Freq uno add to the train dataset
train_inputs.extend(one_freq_input_ids)
train_labels.extend(one_freq_labels)
train_masks.extend(one_freq_attention_masks)
train_token_types.extend(one_freq_token_types)

# pasar a formato torch
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_token_types = torch.tensor(train_token_types)

validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_token_types = torch.tensor(validation_token_types)

# definicion tamaño batch
batch_size = 8


# Pasar a DataLoader

train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# guardar los conjuntos de prueba y de validacion
torch.save(validation_dataloader,'validation_data_loader_transv')
torch.save(train_dataloader,'train_data_loader_transv')

# modelo con el finetuning agregado

#model = AlbertForSequenceClassification.from_pretrained("albert-base-v2", num_labels=num_labels)
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=num_labels)
#model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
#model = RobertaForSequenceClassification.from_pretrained("roberta-large", num_labels=num_labels)
model.to(device)

# parametros.
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# optimizador
optimizer = AdamW(optimizer_grouped_parameters,lr=2e-5,correct_bias=True)


train_loss_set = []

# Numero de epocas
epochs = 300
bestAccu = 0
bestF1 = 0
bestModel = 0

# Train
for _ in trange(epochs, desc="Epoch"):

  # Modelo modo train
  model.train()

  # Tracking variables
  tr_loss = 0 #running loss
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels, b_token_types = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    loss_func = BCEWithLogitsLoss(pos_weight=pesos.to(device)) 
    loss = loss_func(logits.view(-1,num_labels),b_labels.type_as(logits).view(-1,num_labels)) 
    train_loss_set.append(loss.item())

    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    # scheduler.step()
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))

###############################################################################

  # Validation

  # modo Validacion
  model.eval()

  # Variables to gather full output
  logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

  # Predict
  for i, batch in enumerate(validation_dataloader):
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels, b_token_types = batch
    with torch.no_grad():
      # Forward pass
      outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      b_logit_pred = outs[0]
      pred_label = torch.sigmoid(b_logit_pred)

      b_logit_pred = b_logit_pred.detach().cpu().numpy()
      pred_label = pred_label.to('cpu').numpy()
      b_labels = b_labels.to('cpu').numpy()

    tokenized_texts.append(b_input_ids)
    logit_preds.append(b_logit_pred)
    true_labels.append(b_labels)
    pred_labels.append(pred_label)

  # Flatten outputs
  pred_labels = [item for sublist in pred_labels for item in sublist]
  true_labels = [item for sublist in true_labels for item in sublist]

  # Calculate Accuracy
  threshold = 0.6
  pred_bools = [pl>threshold for pl in pred_labels]
  true_bools = [tl==1 for tl in true_labels]
  val_f1_accuracy = f1_score(true_bools,pred_bools,average='micro')*100
  val_flat_accuracy = accuracy_score(true_bools, pred_bools)*100
  if val_f1_accuracy > bestF1:
      bestF1 = val_f1_accuracy
  if val_flat_accuracy > bestAccu:
      bestAccu = val_flat_accuracy
  modelPerf = (val_f1_accuracy+val_flat_accuracy)/2
  if modelPerf > bestModel:
      bestModel = modelPerf
      # guardar modelo 
      
      torch.save(model, 'Modelos/bert_model_transversales_fix.pth')
  
  print('F1 Validation Accuracy: ', val_f1_accuracy)
  print('Flat Validation Accuracy: ', val_flat_accuracy)

#guardar modelo
#torch.save(model.state_dict(), 'bert_model_transversales')



# prediccion
  
# cargar modelo
model = torch.load('bert_model_transversales.pth')

test_df = pd.read_excel('INC-0108-1208.xlsx')
test_df = test_df[test_df['VICEPRESIDENCIA']== 'GERENCIAS VPEO']
test_df= test_df[pd.notnull(test_df['EVENT_DESC'])]
test_df= test_df[pd.notnull(test_df['INC_DESC'])]
test_df['EVENT_DESC']= test_df['EVENT_DESC'].astype(str)
test_df['INC_DESC']= test_df['INC_DESC'].astype(str)
test_df['texto'] = test_df['EVENT_DESC']+test_df['INC_DESC']
test_df= test_df[pd.notnull(test_df['texto'])]
test_df.reset_index(inplace = True, drop = True)


print('Null values: ', test_df.isnull().values.any())
test_df.head()

test_comments = list(test_df.texto.values)


# Encoding input data
test_encodings = tokenizer.batch_encode_plus(test_comments,max_length=max_length,pad_to_max_length=True)
test_input_ids = test_encodings['input_ids']
test_token_type_ids = test_encodings['token_type_ids']
test_attention_masks = test_encodings['attention_mask']


# Tensores
test_inputs = torch.tensor(test_input_ids)
test_masks = torch.tensor(test_attention_masks)
test_token_types = torch.tensor(test_token_type_ids)

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
# Converting flattened binary values to boolean values

pred_bools = [pl>0.6 for pl in pred_labels] #boolean output after thresholding
test_df_bool = pd.DataFrame(data = pred_bools, columns = label_cols)
test_df_prob = pd.DataFrame(data = pred_labels, columns = label_cols)
test_df_int = test_df_bool.astype(int)
test_df_concat = pd.concat([test_df, test_df_int], axis =1)
test_df_concat_prob = pd.concat([test_df, test_df_prob], axis = 1)

# Print and save classification report
#print('Test F1 Accuracy: ', f1_score(true_bools, pred_bools,average='micro'))
#print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools),'\n')
#clf_report = classification_report(true_bools,pred_bools,target_names=test_label_cols)
#pickle.dump(clf_report, open('classification_report_pyt.txt','wb')) #save report
#print(clf_report)


# idx2label = dict(zip(range(20),label_cols))
# print(idx2label)

# # Getting indices of where boolean one hot vector true_bools is True so we can use idx2label to gather label names
# true_label_idxs, pred_label_idxs=[],[]
# # for vals in true_bools:
# #   true_label_idxs.append(np.where(vals)[0].flatten().tolist())
# for vals in pred_bools:
#   pred_label_idxs.append(np.where(vals)[0].flatten().tolist())
  

# # Gathering vectors of label names using idx2label
# true_label_texts, pred_label_texts = [], []
# # for vals in true_label_idxs:
# #   if vals:
# #     true_label_texts.append([idx2label[val] for val in vals])
# #   else:
# #     true_label_texts.append(vals)

# for vals in pred_label_idxs:
#   if vals:
#     pred_label_texts.append([idx2label[val] for val in vals])
#   else:
#     pred_label_texts.append(vals)
    

# # Decoding input ids to comment text
# texto = [tokenizer.decode(text,skip_special_tokens=True,clean_up_tokenization_spaces=False) for text in tokenized_texts]

# # Converting lists to df
# Comparacion_df = pd.DataFrame({'Texto': texto, 'pred_labels':pred_label_texts})
# a = multilabel_confusion_matrix(true_bools, pred_bools)
# Comparacion_df.to_excel('ResultadosGRTRiesgosTransversales.xlsx')
# Comparacion_df.head()