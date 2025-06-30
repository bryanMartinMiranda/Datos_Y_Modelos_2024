# DATA ANALYSIS FOR WATER QUALITY PROJECT 
This README file includes a description of the main files used for the final results of the
master degree thesis developed by Bryan Martín Miranda Leal, in which the focusing was water
quality analysis in a potabilization process.

# THE Datos2024.csv FILE
The file with the full database used for training the neural network model is the _Datos2024.csv_ file.
In this file we have a total of 16434 registers, where each one of them is made up of 4 parameters. The 
parameters included are the water quality variables of ORP, chlorine and dissolved oxygen. The fourth 
parameter is the stage of the potabilization process where the specific register was measured.

# THE Datos2024_conLODO.csv FILE
This file contains the full database and also the group of data corresponding to atypical data for this 
process; this group of atypical data was sampled for raining conditions in august of 2024. The file 
contains 18454 registers, where each one of them contains the same 4 parameters from the _Datos2024.csv_ 
file.

# THE estadistica_descriptiva.py FILE
This file contains the descriptive statistic applied to the database, which was developed in the Python
language. In this file the _numpy_ and _matplotlib_ libraries are used for managing and plotting the data.
In addition, the _corr(method='pearson')_ module was used to calculate the correlation coeficients between the
4 parameters analyzed.

# THE main.py FILE
In the **main.py** file the MLPClassifier class from the scikit-learn library is used for training and testing 
a generic neural network with the water quality database. First of all, the **MinMaxScaler(feature_range=())**
class is used for scalling the 3 water quality parameters; later, the **train_test_split()** class is used to 
splitting the database in training and testing groups. Finally, the **MLPClassifier()** class is defined and the 
model is fit, which let us to know the validation score for the model and we can also plot the loss curve to see 
how the error of the model is minimized.
