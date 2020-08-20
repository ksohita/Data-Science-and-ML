import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from word2number import w2n

df=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\interview(multivariate reg).csv')
print(df.head())
# To check of there is any preprocessing reqd(by chacking the nill values)
# check the difference.
print(df.isnull())
print(df.isnull().any())
df['experience']=df['experience'].fillna('zero')
print(df)
av=math.floor(df['test_score'].mean())
#print (av)
df['test_score']=df['test_score'].fillna(av)
print(df)
df['experience'] = df['experience'].apply(w2n.word_to_num)
print(df)
X = df[['experience','test_score','interview_score']].values
y = df['salary'].values
reg=linear_model.LinearRegression()
print(reg.fit(X,y))
print(reg.coef_)
print(reg.intercept_)
df2=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\salary(prediction).csv')
print(df2.head())
#predicting by simply passing the values of 3 parameters in [[]]
pr=reg.predict([[2,9,6]])
print(pr)
# predicting from a another csv file
y_pred=reg.predict(df2)
print(y_pred)
df2['salary']=y_pred
print(df2)
# This dataframe can be store with the predicted salary to csv file by using to_csv() function.


