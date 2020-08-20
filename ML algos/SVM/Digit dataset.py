import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits=load_digits()
print(dir(digits))
X=digits['data']
y=digits['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
model=svm.SVC()
print(model.fit(X_train,y_train))
print(model.score(X_test, y_test))

ypred = model.predict(X_test)
cm=confusion_matrix(y_test,ypred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()