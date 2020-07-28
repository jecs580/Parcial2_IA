
import pandas as pd
#import numpy as np

df = pd.read_csv("./dataset.csv")
print("DATASET: Cáncer de mama Wisconsin")
# PREPROCESAMIENTO
X = df[['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]

y = df.diagnosis.map({"M":1, "B":0})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ESCOGEMOS LOS MODELOS
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()

from sklearn.pipeline import Pipeline

print("\n**************** PIPELINE OPCION 1*****************")
pipemodel = Pipeline([
                    ('KNeighborsClassifier', KNeighborsClassifier())])
pipemodel.fit(X_train,y_train)
Ypred = pipemodel.predict(X_test)
print(Ypred)

print("\n**************** PIPELINE OPCION 2*****************")

lista_modelo=[('k-vecinos',knn),('Regresion Logistica',logreg)]
for indice, (nombre, modelo) in enumerate(lista_modelo):
    print(indice, nombre)
    modelo.fit(X_train,y_train)
    ypred = modelo.predict(X_test)
    print(ypred)
    print(modelo.score(X_test,y_test))
    print()