# IMPORTING ALL THE NECESSARY LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.width',320,)
pd.set_option('display.max_columns',15)

# IMPORTING THE DATASET

data=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\Major Project\Data_Train.csv')
print(data.head(5))

# CLEANING THE DATASET AND ANALYSING MISSING VALUES

print(data.shape)
print(data.info())
print(data.isna().any())
print(data.isnull().sum())
# Percentage of missing data in each row
print((data.isnull().sum() / len(data)) * 100)

# Replacing the null and missing values of 'Seats' column with its average
print(data['Seats'].value_counts())
av_seats=math.floor(data['Seats'].mean())
print(av_seats)
data['Seats']=data['Seats'].fillna(av_seats)
data['Seats'].replace(0.,av_seats,inplace=True)
#print(data['Seats'].unique())

#  Removing the formatting
data['Power']=data['Power'].str.replace(' bhp','')
data['Engine']=data['Engine'].str.replace(' CC','')
data['Mileage']=data['Mileage'].str.replace('kmpl','')
data['Mileage']=data['Mileage'].str.replace('km/kg','')
#print(data)

# Replacing the null and missing values of 'Engine' column with its average and converting them to numeric type.
print(data['Engine'].value_counts())
data['Engine'].replace('null',np.nan,inplace=True)
data['Engine'] = pd.to_numeric(data['Engine'])
av_engine=math.floor(data['Engine'].mean())
print(av_engine)
data['Engine']=data['Engine'].fillna(av_engine)

# Replacing the null and missing values of 'Power' column with its average and converting them to numeric type.
print(data['Power'].value_counts())
data['Power'].replace('null',np.nan,inplace=True)
data['Power'] = pd.to_numeric(data['Power'])
av_power=data['Power'].mean()
print(av_power)
data['Power']=data['Power'].fillna(av_power)
data['Power']=data['Power'].round(2)

# Replacing the null and missing values of 'Mileage' column with its average and converting them to numeric type.
print(data['Mileage'].value_counts())
data['Mileage'].replace('null',np.nan,inplace=True)
data['Mileage']=pd.to_numeric(data['Mileage'])
av_mil=(data['Mileage'].mean())
print(av_mil)
data['Mileage']=data['Mileage'].fillna(av_mil)
data['Mileage'].replace(0.0,av_mil,inplace=True)
data['Mileage']=data['Mileage'].round(2)

print(data.dtypes)
print(data)
print(data.isnull().sum())
print(data.describe())

# Checking the corelation between the features using 'Heatmap'.
print(data.corr(method='pearson'))
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(),cmap='gist_earth',annot=True)
plt.show()
print(data)


# VISUALIZATION OF DATA USING MATPLOT AND SEABORN

fig = plt.figure(figsize=(18,18))
fig.subplots_adjust(hspace=0.3, wspace=0.2)
# g1 shows us that maximum cars are of fuel type 'Diesel' and then 'Petrol', other fuel types are very rarely used.
fig.add_subplot(2,2,1)
g1 = sns.countplot(x='Fuel_Type', data=data)
loc,labels = plt.xticks()
g1.set_xticklabels(labels,rotation=0)
# g2 shows us that transmission is maximum of 'Manual' type.
fig.add_subplot(2,2,2)
g2 = sns.countplot(x='Transmission', data=data)
loc,labels = plt.xticks()
g2.set_xticklabels(labels,rotation=0)
# g3 shows us that cars are maximum of first hand which are been sold.
fig.add_subplot(2,2,3)
g3 = sns.countplot(x='Owner_Type', data=data)
loc,labels = plt.xticks()
g3.set_xticklabels(labels,rotation=90)
# g4 shows us that maximum cars belong to 'Mumbai' and least are from 'Ahmedabad' which are been sold.
fig.add_subplot(2,2,4)
g4 = sns.countplot(x='Location', data=data)
loc,labels = plt.xticks()
g4.set_xticklabels(labels,rotation=90)
plt.show()

fig = plt.figure(figsize=(18,18))
fig.subplots_adjust(hspace=0.3, wspace=0.2)
# p1 shows how price of car varies by Kilometers Driven.
ax1 = fig.add_subplot(2,2,1)
plt.xlim([0, 100000])
p1 = sns.scatterplot(x="Kilometers_Driven", y="Price", data=data)
loc, labels = plt.xticks()
ax1.set_xlabel('Kilometer')
# p1 shows how price of car varies by Mileage of car.
ax2 = fig.add_subplot(2,2,2)
p2 = sns.scatterplot(x='Mileage',y='Price', data=data)
loc, labels = plt.xticks()
ax2.set_xlabel('Mileage(kmpl)')
# p1 shows how price of car varies by Power.
ax3 = fig.add_subplot(2,2,3)
p3 = sns.scatterplot(x='Power',y='Price', data=data)
loc, labels = plt.xticks()
ax3.set_xlabel('Power(bmp)')
# p1 shows how price of car varies by Engine.
ax4 = fig.add_subplot(2,2,4)
p4 = sns.scatterplot(x='Engine',y='Price', data=data)
loc, labels = plt.xticks()
ax4.set_xlabel('Engine(CC)')
plt.show()

fig = plt.figure(figsize=(18,5))
fig.subplots_adjust(hspace=0.3, wspace=0.2)
# p5 shows variation of price with year of purchase
ax5 = fig.add_subplot(1,2,1)
p5 = sns.scatterplot(x="Year", y="Price", data=data,color='orange')
loc, labels = plt.xticks()
ax5.set_xlabel('Years')
# p6 shows variation of price with no. of seats in car
ax6 = fig.add_subplot(1,2,2)
p6 = sns.scatterplot(x="Seats", y="Price", data=data,color='orange')
loc, labels = plt.xticks()
ax6.set_xlabel('Seats')
plt.show()

# ploting the average price of different car brands which shows us that maximum price is of 'Lamborgini' cars
data['brand_name'] = data['Name'].apply(lambda x: str(x).split(" ")[0])
df_brand = pd.DataFrame(data.groupby('brand_name')['Price'].mean())
df_brand.plot.bar()
plt.show()

# Removing the outlier due to 'Kilometer_Driven' column which looks abnormal.
data.drop(data[data['Kilometers_Driven'] >= 6500000].index, axis=0, inplace=True)

# It shows the Skewness and Kurtosis i.e the degree of distortion from the normal
# distribution and the measure of outliers present in the distribution.
print("Skew ", data['Price'].skew())
print("kurt ", data['Price'].kurt())

# Our values show that our data is highly skewed and has high value of Kurtosis so we take log of prices to avoid skew
# and kurt values
data['Price_n'] = np.log1p(data['Price'].values)
print("Skew ", data['Price_n'].skew())
print("kurt ", data['Price_n'].kurt())


# Separating our original dataset into independent and dependent datasets.
# So we don't need Location column for our prediction as the prices don't depend much on it.
X=data.drop(['Price','Location','Price_n','brand_name'],axis='columns')
y=data['Price_n']
print(X)
print(y)

# Using Label Encoding method to handle Categorical data in columns like 'Name','Fuel_type',
# 'Transmission' and 'Owner_Type'.
Name_en=LabelEncoder()
X['Name_n'] = Name_en.fit_transform(X['Name'])
Fuel_en=LabelEncoder()
X['Fuel_Type_n'] = Name_en.fit_transform(X['Fuel_Type'])
Transmission_en=LabelEncoder()
X['Transmission_n'] = Transmission_en.fit_transform(X['Transmission'])
Owner_en=LabelEncoder()
X['Owner_Type_n'] = Name_en.fit_transform(X['Owner_Type'])

X.drop(['Name','Fuel_Type','Transmission','Owner_Type'],axis='columns',inplace=True)
print(X)

# Using Stratified KFold Method
# Choosing the best model among the three.
folds=StratifiedKFold(n_splits=4)
score_lr=cross_val_score(LinearRegression(),X,y)
score_rf=cross_val_score(RandomForestRegressor(n_estimators=200),X,y)
score_dt=cross_val_score(tree.DecisionTreeRegressor(),X,y)
print(score_lr.mean())
print(score_rf.mean())
print(score_dt.mean())
# so mean of the scores for 4 folds in each of the model shows us that 'Random Forest Regression' has the best score.

# Now using Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
print(len(X_train))
print(len(X_test))

# Applying the Random Forest Regression to train our model
rdf=RandomForestRegressor(n_estimators=200)
print(rdf.fit(X_train,y_train))
print(rdf.score(X_test,y_test))
ypred=rdf.predict(X_test)
print(ypred)
print(r2_score(ypred, y_test))

plt.scatter(y_test,ypred,alpha=0.4,color='Green')
plt.xlabel('Actual Data(y_test)')
plt.ylabel('Predicted Data(ypred)')
plt.show()


# NOW CONVERTING TEST DATA INTO SUITABLE FORMAT FOR PREDICTION.

test_data=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\Major Project\Data_Test.csv')
test_data_orig = test_data.copy()
print(test_data.head)
print(test_data.shape)
print(test_data.isnull().sum())
test_data['Power']=test_data['Power'].str.replace(' bhp','')
test_data['Engine']=test_data['Engine'].str.replace(' CC','')
test_data['Mileage']=test_data['Mileage'].str.replace('kmpl','')
test_data['Mileage']=test_data['Mileage'].str.replace('km/kg','')

print(test_data)
test_data['Seats']=data['Seats'].fillna(av_seats)
test_data['Seats'].replace(0.,av_seats,inplace=True)
print(test_data['Seats'].unique())

test_data['Engine'].replace('null',np.nan,inplace=True)
test_data['Engine'] = pd.to_numeric(test_data['Engine'])
test_data['Engine']=test_data['Engine'].fillna(av_engine)
print(data['Engine'].unique())

test_data['Power'].replace('null',np.nan,inplace=True)
test_data['Power'] = pd.to_numeric(test_data['Power'])
test_data['Power']=test_data['Power'].fillna(av_power)
test_data['Power']=test_data['Power'].round(2)
print(data['Power'].unique())

test_data['Mileage']=pd.to_numeric(test_data['Mileage'])
print(data['Mileage'].unique())

print(test_data.dtypes)

print(test_data)

test_data=test_data.drop(['Location'],axis='columns')
Name_en=LabelEncoder()
test_data['Name_n'] = Name_en.fit_transform(test_data['Name'])
Fuel_en=LabelEncoder()
test_data['Fuel_Type_n'] = Name_en.fit_transform(test_data['Fuel_Type'])
Transmission_en=LabelEncoder()
test_data['Transmission_n'] = Transmission_en.fit_transform(test_data['Transmission'])
Owner_en=LabelEncoder()
test_data['Owner_Type_n'] = Name_en.fit_transform(test_data['Owner_Type'])

test_data.drop(['Name','Fuel_Type','Transmission','Owner_Type'],axis='columns',inplace=True)

print(test_data)

# Predicting the prices from our modified dataframe
y_predt=rdf.predict(test_data)
print(y_predt)
y_pred_test=np.expm1(y_predt)
print(y_pred_test)

test_data_orig['salary']=y_pred_test
print(test_data_orig)

#test_data_orig.to_csv(r'C:\Users\KIIT\Desktop\ML data sets\Major Project\predicted_prices.csv',index=False)
