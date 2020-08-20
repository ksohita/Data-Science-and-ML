import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
pd.set_option('display.width',320,)
pd.set_option('display.max_columns',15)
data=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\titanic(decision tree-classification).csv')
print(data.head())
df=data[['Survived','Pclass','Sex','Age','Fare']]
print(df.head())
print(df.shape)

indep=df[['Pclass','Sex','Age','Fare']]
dep=df[['Survived']]
en_Sex=LabelEncoder()
indep['Sex_n'] = en_Sex.fit_transform(indep['Sex'])
indep.drop(['Sex'],axis='columns',inplace=True)
print(indep.head())
print(dep.head())

print(indep.isnull().any())
print(indep['Age'].isna().sum())
av=math.floor(indep['Age'].mean())
print(av)
indep['Age']=indep['Age'].fillna(av)
#indep['Fare']=indep['Fare'].round(3)
print(indep.head(20))

X_train, X_test, y_train, y_test = train_test_split(indep,dep, test_size=0.2, random_state=0)
print(len(X_train))
print(len(X_test))
model = tree.DecisionTreeClassifier( criterion='gini',random_state=0,splitter='best',max_depth=5,min_samples_leaf=10,min_samples_split=10)
#model = tree.DecisionTreeClassifier()
print(model.fit(X_train,y_train))
print(model.score(X_test,y_test))
print(model.predict([[2,34,20.67,0]]))
ypred=model.predict((X_test))

cm=confusion_matrix(y_test,ypred)
plt.figure(figsize=(5,4))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



