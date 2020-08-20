import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

df=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\insurance dataset.csv')
print(df.head())
# here x has double brackets as it can take multi value.
X = df[['age']]
y = df['bought_insurance']
#Splitting the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg=linear_model.LogisticRegression()
print(reg.fit(X_train,y_train))
print(reg.coef_)
print(reg.intercept_)
print(y_test)
y_pred=reg.predict(X_test)
print(y_pred)
print(X_test)

# score tells you the accuracy of the model
print(reg.score(X_test,y_test))
# It gives the probability of testing data to be in 1 or 0
print(reg.predict_proba(X_test))
print(reg.predict([[67]]))
print(reg.predict([[26]]))

plt.scatter(df['age'],df['bought_insurance'],marker='^',color='orange')
# plt.plot(df.year,reg.predict(df[['year']]),color='red')
plt.xlabel('Age')
plt.ylabel('bought_insurance')
plt.show()