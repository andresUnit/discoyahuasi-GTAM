# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 16:19:10 2020

author: AMS
"""


import pandas as pd

df1 = pd.read_excel("Data Clasificada/clasificación GRTs-GTAM-Operaciones TAM  1A.xlsx", header = 0)

labels = df1.keys()[-55:-26]

df1.dropna(subset=labels, thresh=1, inplace=True)

df1.replace(["X"],1, inplace = True)

df2 = pd.read_excel("Data Clasificada/clasificación GRTs-GTAM-Operaciones TAM  1B.xlsx", header = 1)

df2.dropna(subset=labels, thresh = 1, inplace=True)

df3 = pd.read_excel("Data Clasificada/clasificacioìn GRTs-GTAM-Infra y trans de fluidos 2B.xlsx", sheet_name= "A presentar.", header=8)

df3.dropna(subset=labels, thresh=1, inplace=True)

df4 = pd.read_excel("Data Clasificada/clasificación GRTs-GTAM-Infra y trans de fluidos 2A.xlsx", sheet_name= "A presentar.")
df4.dropna(subset=labels, thresh=1, inplace=True)


df_aux =pd.merge(df1,df2,how="outer")
df = pd.merge(df_aux,df3, how = "outer")
df = pd.merge(df,df4, how =  "outer")
df['texto'] = df['EVENT_DESC'] + df['INC_DESC']+df['DESVC6_DESCRIPCION'].fillna('')+df['DESVC1_DESCRIPCION'].fillna('')+df['DESVC2_DESCRIPCION'].fillna('')+df['DESVC3_DESCRIPCION'].fillna('')+df['DESVC4_DESCRIPCION'].fillna('')+df['DESVC5_DESCRIPCION'].fillna('')

df.drop_duplicates(subset=['EVENT_NUMBER'], keep='first', inplace=True, ignore_index=False)
df.drop_duplicates(subset=['texto'], keep='first', inplace=True, ignore_index=False)
df.dropna(subset=['texto'], inplace=True)

allkeys= df.keys()
dropKeys = allkeys[-53:-1]

df.drop(dropKeys, axis=1, inplace=True)

df.drop(['Trabajos fuera de faena, zona limítrofe', 'Afectación ambiental', 'Interacción con las comunidades'], axis=1, inplace= True)

df.fillna(0, inplace = True)


df["Carga, transporte y descarga"]=(df["Faltan controles en estiba de carga"].astype("int64") | df["Deficiente Carga, transporte y descarga "].astype("int64"))

df.drop(["Faltan controles en estiba de carga","Deficiente Carga, transporte y descarga "], axis=1, inplace = True)

df["Maniobras de izaje"]=(df["Plan de izaje deficiente "].astype("int64") | df["Operación  deficiente de equipos de levante "].astype("int64"))

df.drop(["Plan de izaje deficiente ","Operación  deficiente de equipos de levante "], axis=1, inplace = True)

df["Retraso en inicio de los trabajos"]=(df["Retraso en incio de los trabajos"].astype("int64") | df["Falta de dotación"].astype("int64"))

df.drop(["Retraso en incio de los trabajos","Falta de dotación"], axis=1, inplace = True)

df["Energía de orígen y potencial no identificada "]=(df["Energía de origen y potencial no identificada"].astype("int64") | df["Exposición a descarga eléctrica "].astype("int64"))

df.drop(["Energía de origen y potencial no identificada","Exposición a descarga eléctrica "], axis=1, inplace = True)

df["Entorno de trabajo no controlado "]=(df["Deficiente evaluación del entorno"].astype("int64") | df["Entorno de trabajo no controlado"].astype("int64") | df["Superficies de trabajo irregular "].astype("int64"))

df.drop(["Deficiente evaluación del entorno","Entorno de trabajo no controlado","Superficies de trabajo irregular "], axis=1, inplace = True)

df["Conducción"]=(df["Conducción interior faena"].astype("int64") | df["Conducción exterior  faena"].astype("int64"))

df.drop(["Conducción interior faena","Conducción exterior  faena"], axis=1, inplace = True)

df.to_excel('Base Datos/data_GTAM_fix.xlsx', index=False)

