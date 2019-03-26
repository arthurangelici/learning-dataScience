# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:31:01 2019

@author: arthur
"""

import pandas as pd

base = pd.read_csv('risco-credito.csv')
previsores = base.iloc[:,0:4].values
df = pd.DataFrame(previsores)
classe = base.iloc[:,4].values
dp = pd.DataFrame(classe)

from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
previsores[:,0] =labelEncoder.fit_transform(previsores[:,0])
previsores[:,1] =labelEncoder.fit_transform(previsores[:,1])
previsores[:,2] =labelEncoder.fit_transform(previsores[:,2])
previsores[:,3] =labelEncoder.fit_transform(previsores[:,3])

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores, classe)

#novo input prever conforme modelo teorico - esperado baixo
#historia boa, dívida alta, garantias nenhuma, renda>35
resultado = classificador.predict([[0,0,1,2]])

#historia ruim, divida alta, garantias adequada, renda <15
resultado = classificador.predict([[3,0,0,0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)