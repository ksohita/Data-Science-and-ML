import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


pd.set_option('display.width',320,)
pd.set_option('display.max_columns',10)
df = pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\car model(get dummy-reg with categ).csv')
print(df.head())

x=df['Mileage']
y=df['Sell Price($)']
age=df['Age(yrs)']

# So this graph shows us that our price depends on mileage and age so Linear Reg Model can be used.
plt.scatter(x,y,c=age,marker='^',cmap='magma',s=50)
plt.xlabel('Distance travelled')
plt.ylabel('Sell Price($)')
plt.title('Comparison of Mileage with sell price')
cbar=plt.colorbar()
cbar.set_label('Age')
plt.tight_layout()
plt.show()

# To check if null values exist?
print(df.isnull().any())

dummies=pd.get_dummies(df['Car Model'])
print(dummies)
merged=pd.concat([df,dummies], axis='columns')
print(merged)
merged.drop(['Mercedez Benz C class','Car Model'],axis='columns',inplace=True)
print(merged)

yvar=merged['Sell Price($)'].values
xvar=merged[['Mileage','Age(yrs)','Audi A5','BMW X5']].values

print(xvar)
print(yvar)
X_train, X_test, y_train, y_test = train_test_split(xvar, yvar, test_size=0.2, random_state=0)
reg=linear_model.LinearRegression()
print(reg.fit(X_train,y_train))
print(reg.predict([[4500,4,0,0]]))
print(reg.score(X_test,y_test))
y_pred=reg.predict(X_test)
print(X_test)
print(y_pred)
print(y_test)



