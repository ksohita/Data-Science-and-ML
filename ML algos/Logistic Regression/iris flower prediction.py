import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import seaborn as sn

iris = load_iris()
print(dir(iris))

print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(iris.filename)

plt.scatter(iris.data[:,0],iris.data[:,1],c=iris.target,s=20,marker="o")
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

plt.scatter(iris.data[:,2],iris.data[:,3],c=iris.target,s=20,marker="o")
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'],test_size=0.2, random_state=0)
print(len(X_test))
print(len(X_train))
# here we need to increse no. of itereations as its default value is 1000)
reg = linear_model.LogisticRegression(solver='liblinear',max_iter=2000)
print(reg.fit(X_train, y_train))
print(reg.score(X_test,y_test))
ypred=reg.predict(X_test)
print(ypred)
prob=reg.predict_proba(X_test)
print(prob)

cm=confusion_matrix(y_test,ypred)
plt.figure(figsize=(5,4))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

