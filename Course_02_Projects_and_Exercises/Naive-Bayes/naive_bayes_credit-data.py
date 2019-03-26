# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:21:35 2019

@author: arthur
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 15:23:26 2019

@author: arthur
"""

import pandas as pd

base = pd.read_csv('credit-data.csv')
base.loc[base.age<0, 'age'] = 40.9277

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matrix = confusion_matrix(classe_teste, previsoes)
