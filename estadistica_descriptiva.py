import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar archivo CSV.
datosProceso=pd.read_csv('Datos2024.csv')

Pd_ORP=datosProceso.ORP
Pd_DO=datosProceso.DO
Pd_Cloro=datosProceso.cloroResidual
Pd_target=datosProceso.fase

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
