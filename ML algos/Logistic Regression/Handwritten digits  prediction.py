import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics

digits = load_digits()
print(dir(digits))

print(digits.data[0])
print(digits.images[0])
#plt.gray()
print(digits.target[60:65])
print(digits.target_names[60:65])

for i in range(60,65,1):
    plt.matshow(digits.images[i])
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits['data'],digits['target'],test_size=0.2, random_state=0)
print(len(X_test))
print(len(X_train))
# here we need to increse no. of itereations as its default value is 1000)
reg = linear_model.LogisticRegression(solver='liblinear',max_iter=2000)
print(reg.fit(X_train, y_train))
print(reg.score(X_test,y_test))
plt.matshow(digits.images[965])
plt.show()
print(reg.predict(digits.data[[965]]))

ypred=reg.predict(X_test)
cm=confusion_matrix(y_test,ypred)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, ypred))