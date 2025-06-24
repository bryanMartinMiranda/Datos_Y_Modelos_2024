# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 23:35:55 2025

@author: bryan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar los CSV
datosProceso=pd.read_csv('Datos2024.csv')

# PANDAS
Pd_ORP=datosProceso.ORP
Pd_DO=datosProceso.DO
Pd_Cloro=datosProceso.cloroResidual
Pd_target=datosProceso.fase

# NUMPY
Np_DO=np.array([datosProceso.DO])
Np_ORP=np.array([datosProceso.ORP])
Np_Cloro=np.array([datosProceso.cloroResidual])
Np_target=np.array([datosProceso.fase])

# DESCRIPCIÓN PUNTUAL
print('Promedios >>> ','ORP= ',np.mean(Np_ORP),
                       'DO= ',np.mean(Np_DO),
                       'cloro= ',np.mean(Np_Cloro),'\n')
print('Valores Maximos >>> ','ORP= ',np.max(Np_ORP),
                             'DO= ',np.max(Np_DO),
                             'cloro= ',np.max(Np_Cloro),'\n')
print('Valores Minimos >>> ','ORP= ',np.min(Np_ORP),
                             'DO= ',np.min(Np_DO),
                             'cloro= ',np.min(Np_Cloro),'\n')

datosCorrelacion=datosProceso[['ORP','cloroResidual','DO']]

# CORRELACIÓN
print('\nMatriz de correlacion: \n')
print(datosCorrelacion.corr(method='pearson'))

#datosDiagramaCaja=datosCorrelacion.to_numpy()

# DIAGRAMAS DE CAJA

figura1=plt.figure()
plt.boxplot(Np_ORP.T)
plt.title('ORP')
plt.show()

figura2=plt.figure()
plt.boxplot(Np_Cloro.T)
plt.title('Cloro Residual')
plt.show()

figura3=plt.figure()
plt.boxplot(Np_DO.T)
plt.title('Oxígeno Disuelto')
plt.show()

# MLPC SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

oxigeno1=np.array([datosProceso.DO])
ORP1=np.array([datosProceso.ORP])
Cloro1=np.array([datosProceso.cloroResidual])
target1=np.array([datosProceso.fase])

scaler1=MinMaxScaler(feature_range=(0.01, 1))
scaler1.fit(oxigeno1.T)
oxigenoEscalado1=scaler1.transform(oxigeno1.T)

scaler2=MinMaxScaler(feature_range=(0.01,1))
scaler2.fit(ORP1.T)
orpEscalado1=scaler2.transform(ORP1.T)

scaler3=MinMaxScaler(feature_range=(0.01,1))
scaler3.fit(Cloro1.T)
CloroEscalado1=scaler3.transform(Cloro1.T)

oxigenoOrdenado1=oxigenoEscalado1.reshape(16434,)
OrpOrdenado1=orpEscalado1.reshape(16434,)
cloroOrdenado1=CloroEscalado1.reshape(16434,)
targetsOrdenados1=target1.reshape(16434,)
datasetNumpyScaled=np.column_stack((OrpOrdenado1,
                                    cloroOrdenado1,
                                    oxigenoOrdenado1,
                                    targetsOrdenados1))

parametros1=datasetNumpyScaled[:,0:3]
objetivos1=datasetNumpyScaled[:,3]
X_train,X_test,y_train,y_test=train_test_split(parametros1,objetivos1,train_size=0.8,random_state=10,shuffle=True)

modelo1=MLPClassifier(hidden_layer_sizes=(30,),activation='logistic',solver='sgd',learning_rate='constant',
                      learning_rate_init=0.02,max_iter=80,random_state=(10))
modelo1.fit(X_train,y_train)
print(modelo1.n_iter_)
print('\n')
print(modelo1.n_layers_)
print('\n')
print(modelo1.n_outputs_)
print('\n')
print(modelo1.out_activation_)
print('Porcentaje de aciertos:',modelo1.score(X_test, y_test))
