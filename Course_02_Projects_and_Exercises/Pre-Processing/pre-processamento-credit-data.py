# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:23:26 2019

@author: arthur
"""

import pandas as pd

base = pd.read_csv('credit-data.csv')
#problema de valor inconsistente - idade negativa
#solução - preencher os valores com a média
base.loc[base.age<0, 'age'] = 40.9277

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

#problema de valor nulo
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

#escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

#divisão dataset treino e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

