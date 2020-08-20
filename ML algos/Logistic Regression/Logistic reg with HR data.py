import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

pd.set_option('display.width',320,)
pd.set_option('display.max_columns',15)
df=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\HR_comma_sep.csv')
print(df.head())
print(df.shape)
data1=df.groupby('left').mean()
print(data1)

print(df['salary'].unique())
salary=['low','medium','high']
xpos=np.arange(len(salary))
w=0.25
dataset = df['left'].groupby([df['salary'], df['left']]).count().unstack()
print(dataset)
y_count=np.arange(0,6000,1000)
stay=list(dataset[0])
left=list(dataset[1])
plt.bar(xpos+w/2,stay,label=0,color='orange',width=w)
plt.bar(xpos-w/2,left,label=1,color='green',width=w)
plt.xlabel('salary')
plt.ylabel('count of people')
plt.legend()
plt.xticks(ticks=xpos,labels=salary)
plt.yticks(ticks=y_count,labels=y_count)
plt.tight_layout()
plt.show()


dummies=pd.get_dummies(df['salary'])
print(dummies)
merged=pd.concat([df,dummies], axis='columns')
merged=merged.drop(['low','salary'], axis='columns')
print(merged)
X=merged[['satisfaction_level','average_montly_hours','promotion_last_5years','medium','high']]
y=merged['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg=linear_model.LogisticRegression(solver='solver')
print(reg.fit(X_train,y_train))

y_pred=reg.predict(X_test)
print(y_pred)
print(y_test)
print(reg.score(X_test,y_test))
prob=reg.predict_proba(X_test)

print(reg.predict([[0.67,250,1,1,0]]))
