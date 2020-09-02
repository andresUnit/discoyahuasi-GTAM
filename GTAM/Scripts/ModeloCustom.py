# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:45:51 2020

author: AMS
"""


# Imports
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
import torch.nn.functional as F
import torch.optim as optim


# device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

focos = ['Defciente controles  plan Covid 19', 'Falta internalización del CGR', 'Ausencia o deficiencia plan específico de bloqueo', 'VATS deficiente', 'Ejecución deficiente de la actividad ', 'Instructivo  deficiente', 'Ausencia de planificación', 'Controles  no aplicados correctamente', 'Trabajos en caliente', 'Orden y aseo deficiente en área de trabajo', 'Residuos industriales  peligrosos -domésticos', 'Falta de stock de respuestos', 'Carga, transporte y descarga', 'Maniobras de izaje', 'Retraso en inicio de los trabajos', 'Energía de orígen y potencial no identificada ', 'Entorno de trabajo no controlado ', 'Conducción']

df = pd.read_excel("Base Datos/Data Post Balance Bert/Aprendizaje de evento anterior no internalizado.xlsx")

for foco in focos:
    aux = pd.read_excel(f"Base Datos/Data Post Balance Bert/{foco}.xlsx")
    df = pd.concat([df,aux])
    
df.fillna(0, inplace = True)
df.reset_index(inplace=True, drop=True)

cols = df.columns
label_cols = list(cols[-19:])
num_labels = len(label_cols)
print('Label columns: ', label_cols)

print('Count of 1 per label: \n', df[label_cols].sum(), '\n')
print('Count of 0 per label: \n', df[label_cols].eq(0).sum())

df[label_cols].sum().sort_values().plot.barh(grid = True, title = 'Gráfico Distribución Labels Focos Transversales', xlim = (0,140))

pesos = torch.tensor(df[label_cols].sum().max()/df[label_cols].sum()[label_cols].values)

#shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Generacion de los labels formato One Hot
df['one_hot_labels'] = list(df[label_cols].values)
df.head()

# Generacion de Data a Utilizar

labels = list(df.one_hot_labels.values)

data = df.loc[:,:'Defciente controles  plan Covid 19']
data.drop(['Defciente controles  plan Covid 19'], axis = 1, inplace = True)
data = data.values
data = data.tolist()

# frecuencia uno
label_counts = df.one_hot_labels.astype(str).value_counts()
print('Freq: \n', label_counts)

# encontramos el indice y retornamos
one_freq = label_counts[label_counts==1].keys()
one_freq_idxs = sorted(list(df[df.one_hot_labels.astype(str).isin(one_freq)].index), reverse=True)
print('df label indices with only one instance: ', one_freq_idxs)

# extraemos el valor y lo guardamos
one_freq_input = [data.pop(i) for i in one_freq_idxs]
one_freq_labels = [labels.pop(i) for i in one_freq_idxs]


# Data Split

train, validation, train_labels, validation_labels= train_test_split(data, labels,random_state=2020, test_size=0.20, stratify = labels)

# Freq uno add to the train dataset
train.extend(one_freq_input)
train_labels.extend(one_freq_labels)

# pasar a formato torch
train = torch.tensor(train)
train_labels = torch.tensor(train_labels)


validation = torch.tensor(validation)
validation_labels = torch.tensor(validation_labels)

# definicion tamaño batch
batch_size = 8

# Pasar a DataLoader

train_data = TensorDataset(train, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# modelo Finetuning

class Clasificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear1 = nn.Linear(768, 394)
        self.Linear2 = nn.Linear( 394,19)
        #self.Linear3 = nn.Linear(394, 19)
        self.Dropout1 = nn.Dropout(0.1)
        #self.Dropout2 = nn.Dropout(0.1)
        
    def forward(self, input):
        x = self.Dropout1(input)
        x = F.relu(self.Linear1(x))
        #x = self.Dropout2(x)
        x = self.Linear2(x)
        #x = F.softmax(self.Linear3(x))
        return x

model = Clasificador()
model.to(device)


# optimizador
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loss_set = []

# Numero de epocas
epochs = 5000
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
    b_input_ids, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    outputs = model(b_input_ids)
    logits = outputs
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
    b_input_ids, b_labels= batch
    with torch.no_grad():
      # Forward pass
      outs = model(b_input_ids)
      b_logit_pred = outs
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
  threshold = 0.5
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
      
      torch.save(model, 'Modelos/bert_model_transversales_fix_clasificador.pth')
  
  print('F1 Validation Accuracy: ', val_f1_accuracy)
  print('Flat Validation Accuracy: ', val_flat_accuracy)
  
  
model = torch.load('Modelos/bert_model_transversales_fix_clasificador.pth')


grt = pd.read_excel("Inferencias/test_inc_fix.xlsx")
test_inputs = grt.values
test_inputs = test_inputs.tolist()

# Tensores
test_inputs = torch.tensor(test_inputs)

# Create test dataloader
test_data = TensorDataset(test_inputs)# test_labels,
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

model.eval()

#track variables
logit_preds,pred_labels,tokenized_texts = [],[],[]

# Predict
for i, batch in enumerate(test_dataloader):
  # Unpack the inputs from our dataloader
  b_input_ids = batch[0]
  with torch.no_grad():
    # Forward pass
    outs = model(b_input_ids.to(device))
    b_logit_pred = outs
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

pred_bools = [pl>0.5 for pl in pred_labels] #boolean output after thresholding
test_df_bool = pd.DataFrame(data = pred_bools, columns = label_cols)
test_df_prob = pd.DataFrame(data = pred_labels, columns = label_cols)
test_df_int = test_df_bool.astype(int)
test_df_int.to_excel("auxinc.xlsx",index = False)







