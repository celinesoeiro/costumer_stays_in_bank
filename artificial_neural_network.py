# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
# Separando as variáveis independentes 
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder # Usou o oneHotEncoder porque a ordem não importa
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
#X = X[:,1:] # para fugir da dummy variable trap - NÃO PRECISOU!

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential() # Permite criar uma sequência de camadas

# Adding the input layer and the first hidden layer
# units = (11 + 1)/2, número de variáveis independentes / 2 (units é o número de neuronios da camada escondida)
# input_dim é o número de neurônios da primeira camada
# units é um número escolhido empiricamente. Só testando pra saber o melhor valor. Não existe uma regra específica
# activation = relu -> relu = rectifier function
# hyper paramets: units, activation
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim = 12))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
# saída tem de ser binária, por isso sigmoid function
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
# batch_size -> Treinar com o método batch gradient descent (batch learning) produz melhores resultados
# que treinar com o estocástico. Pois comparar o valor total da probabilidade é mais eficaz
# que comparar um por um. 
# o numero 32 é um número mágico classico, sempre utilizado, e indica o nº de predições 
# que se quer obter para comparar com o nº de valores reais.
# epochs: nº de vezes que passa por todos os dados
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
right = accuracy_score(y_test, y_pred)
# right = (cm[0,0] + cm[1,1])/len(y_test) mesma que a função acima (accuracy_score)
print(right*100)

# Predicting the result of a single observation

# Geography: France
# Credit Score: 600
# Geder: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: 60000
# Number of products: 2
# Credit Card: yes
# Active Member: yes
# Estimated Salary: 50000

# Leaving?

X_desired = [[1,0,0,600,1,40,3,60000,2,1,1,50000]];
X_desired_scaled = sc.fit_transform(X_desired)
print(X_desired_scaled)

y_desired = ann.predict(X_desired_scaled)
print(y_desired[0])
if (y_desired[0] < 0.5):
    print('Fica no banco')
else:
    print('Sai do banco')