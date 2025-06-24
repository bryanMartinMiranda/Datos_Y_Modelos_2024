# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 18:26:23 2025

@author: bryan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
random.seed(10)
np.random.seed(10)

# Importar los CSV
datosProceso=pd.read_csv('Datos2024.csv')

# PANDAS
Pd_ORP=datosProceso.ORP
Pd_Cloro=datosProceso.cloroResidual
Pd_Oxigeno=datosProceso.DO
Pd_target=datosProceso.fase

# NUMPY
Np_ORP=np.array([datosProceso.ORP])
Np_Cloro=np.array([datosProceso.cloroResidual])
Np_Oxigeno=np.array([datosProceso.DO])
Np_target=np.array([datosProceso.fase])

# MLPC SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve

ORP1=np.array([datosProceso.ORP])
Cloro1=np.array([datosProceso.cloroResidual])
Oxigeno1=np.array([datosProceso.DO])
target1=np.array([datosProceso.fase])

scaler1=MinMaxScaler(feature_range=(0.01,1))
scaler1.fit(Oxigeno1.T)
oxigenoEscalado1=scaler1.transform(Oxigeno1.T)

scaler2=MinMaxScaler(feature_range=(0.01,1))
scaler2.fit(ORP1.T)
orpEscalado1=scaler2.transform(ORP1.T)

scaler3=MinMaxScaler(feature_range=(0.01,1))
scaler3.fit(Cloro1.T)
CloroEscalado1=scaler3.transform(Cloro1.T)

OrpOrdenado1=orpEscalado1.reshape(16434,)
cloroOrdenado1=CloroEscalado1.reshape(16434,)
oxigenoOrdenado1=oxigenoEscalado1.reshape(16434,)
targetsOrdenados1=target1.reshape(16434,)

datasetNumpyScaled=np.column_stack((OrpOrdenado1,
                                    cloroOrdenado1,
                                    oxigenoOrdenado1,
                                    targetsOrdenados1))

parametros1=datasetNumpyScaled[:,0:3]
objetivos1=datasetNumpyScaled[:,3]
X_train,X_test,y_train,y_test=train_test_split(parametros1,objetivos1,train_size=0.7,random_state=10,shuffle=True)

modelo1=MLPClassifier(hidden_layer_sizes=(15,),activation='logistic',solver='sgd',learning_rate='constant',
                      learning_rate_init=0.25,max_iter=800,random_state=(10))

modelo1.fit(X_train,y_train)

plt.plot(modelo1.loss_curve_,'r')
plt.title('Curva de aprendizaje')
plt.xlabel('Épocas')
plt.ylabel('Loss')
#plt.show()
#plt.savefig('picture.png',dpi=500,format='png')

print(modelo1.n_layers_)
print(modelo1.out_activation_)
print('Iteraciones:',modelo1.n_iter_)
print('Porcentaje de aciertos:',modelo1.score(X_test, y_test))
