import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



pd.set_option('display.width',320,)
pd.set_option('display.max_columns',10)
df = pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\car model(get dummy-reg with categ).csv')
print(df.head())
y=df['Sell Price($)']
X=df[['Car Model','Mileage','Age(yrs)']]
carmod_le=LabelEncoder()
X['Car Modeln']= carmod_le.fit_transform(X['Car Model'])
X.drop(['Car Model'],axis='columns',inplace=True)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))
model = RandomForestRegressor(n_estimators=200,random_state=0)
print(model.fit(X_train,y_train))
print(model.score(X_test,y_test))
print(model.predict(X_test))
print(y_test)
