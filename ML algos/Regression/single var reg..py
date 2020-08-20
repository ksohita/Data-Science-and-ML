import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
df=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\per capita income.csv')
print(df.head())
# use reshape(-1,1) when only there is only 1 independent variable
X = df['year'].values.reshape(-1,1)
y = df['per capita income (US$)'].values.reshape(-1,1)
# when random_state set to an integer, train_test_split will return same results for each execution.
# when random_state set to an None, train_test_split will return different results for each execution.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg=linear_model.LinearRegression()
print(reg.fit(X_train,y_train))
print(reg.coef_)
print(reg.intercept_)

plt.scatter(df['year'],df['per capita income (US$)'],marker='*',color='green')
plt.plot(df.year,reg.predict(df[['year']]),color='red')
plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.show()

#df2=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\years_prediction.csv')
#print(df2.head())
p=reg.predict(X_test)
print(p)
print(y_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, p))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, p))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, p)))
print(reg.predict([[2020]]))
#df2['per_capita_income']=p
#print(df2)
# to convert the new dataframe into csv file
#df2.to_csv(r'C:\Users\KIIT\Desktop\ML data sets\prediction(linear).csv',index=False)
