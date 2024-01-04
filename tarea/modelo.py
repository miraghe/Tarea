from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

iris = datasets.load_iris()

X = iris.data[:,[2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 1, stratify = y)


sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# Guardamos los datos base, los nombres de las flores y los datos de escalado

with open('tarea/datos/base.pck', 'wb') as f:
    pickle.dump((list(iris.target_names), sc), f)


# Regresión logistica
    
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')
model.fit(X_train_std, y_train)

with open('tarea/datos/regresion.pck', 'wb') as f:
    pickle.dump(model, f)


# SVM
    
from sklearn.svm import SVC

model = SVC(kernel='linear',C=1.0, random_state=1, probability=True) # probability=True es importante para poder usar predict_proba

model.fit(X_train_std, y_train)

with open('tarea/datos/svm.pck', 'wb') as f:
    pickle.dump(model, f)


# Árbol de decisión
    
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',max_depth=4, random_state=1)

model.fit(X_train_std, y_train)

with open('tarea/datos/arbol.pck', 'wb') as f:
    pickle.dump(model, f)


# K-NN
    
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

model.fit(X_train_std, y_train)

with open('tarea/datos/knn.pck', 'wb') as f:
    pickle.dump(model, f)
