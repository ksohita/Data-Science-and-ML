import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import seaborn as sn

iris = load_iris()
print(dir(iris))

X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'],test_size=0.2, random_state=0)
print(len(X_test))
print(len(X_train))

model = RandomForestClassifier(random_state=0,n_estimators=100,criterion='entropy')
print(model.fit(X_train, y_train))
print(model.score(X_test,y_test))
ypred=model.predict(X_test)
print(ypred)
print(y_test)
cm=confusion_matrix(y_test,ypred)
plt.figure(figsize=(5,4))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


