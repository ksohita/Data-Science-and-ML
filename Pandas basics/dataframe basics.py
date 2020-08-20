import pandas as pd

# PERFORMING BASIC OPERATIONS IN A DATAFRAME:

data=pd.read_csv(r'C:\Users\KIIT\Desktop\ML data sets\test.txt', sep=';')
# to add index instead of default index values 0,1,2...
data.index=["DL","JAI","DEH","MUM","NOI","LUCK","SRI","MUS"]
print(data)
# It is of dataframe type
print(type(data))
# shape attribute tells the total rows and columns in dataframe
print(data.shape)
# The describe() method is used for calculating some statistical data of the numerical
# values of the Series or DataFrame.
print(data.describe())
# The info() function is used to print a concise summary of a DataFrame.
print(data.info())
print(data.head(1))
print(data.tail(3))
# prints the particular columns of the data frame
print(data[['city','mintemp']])
print(type(data[['city','mintemp']]))
# Datatype of a single coloumn is a series
print(type(data['city']))
# prints list of all columns in the dataframe
print(data.columns)
# iloc(indexed loc) is used to fetch the particular rows in particular columns by specifying the dafault index values
print(data.iloc[[1,4],[1,2,3]])
# slicing where end point is not included (in iloc)
print(data.iloc[0:2,1:3])
# loc is used to fetch the particular rows in particular columns by specifying the given index values
print(data.loc[['JAI','DEH'],['day','city','mintemp']])
# slicing where end point is included (in loc)
print(data.loc['DL':'MUM','mintemp':'humidity'])
# counts no. of uniques values of each row for particular column
print(data['day'].value_counts())
# Total entries in 'day' column
print(data['day'].count())
# Gives list of all unique values for that column
print(data['day'].unique())
# gives no. of sll unique values for that column
print(data['day'].nunique())
# set the index as one of its provided columns and inplace=True marks permanent change
data.set_index('city',inplace=True)
print(data)
# To print all the index values
print(data.index)
# it resets the index values to default 0,1,2...values
print(data.reset_index(inplace=True))
print(data)
# Just setting our original index values
data.index=["DL","JAI","DEH","MUM","NOI","LUCK","SRI","MUS"]
print(data)
# prints the data after sorting the index values.
print(data.sort_index())

# FILTERING DATA:

# Returns all indexes with boolean values as true anf false for given conditions
print(data['day']=='Sunday')
# With this,it gives a better result as compared to previous one as it returns the entire dataframe which has true condiion
# True condition
flt=(data['day']=='Sunday')
print(data[flt])
# gives same result as above,but we can here specify the particular columns required as done below(mostly used)
print(data.loc[flt])
print(data.loc[flt,['mintemp','humidity']])
# Returns the rows with the 2 given condition with AND operation b/w them
flt=(data['day']=='Sunday') & (data['windy']==23)
print(data.loc[flt])
# Returns the rows with the 2 given condition with OR operation b/w them
flt=(data['day']=='Sunday') | (data['maxtemp']==36)
print(data.loc[flt])
# returns opposite of the condition
print(data.loc[~flt])

# Returns all ciites wich comes in this list of cities because here if we apply OR multiple times
# it will make it large enough
# cities=['Delhi','Jaipur','Mumbai','Noida']
flt=(data['city'].isin(['Delhi','Jaipur','Mumbai','Noida']))
print(data.loc[flt,['day','city']])
# Returns all those cities which contains 'D'
flt=(data['city'].str.contains('D'))
print(data.loc[flt,['day','city']])

# UPDATING, ADDING AND DELETING ROWS AND COLUMNS

# To give new names to all columns
data.columns=['City','Day','Min temp','Max temp','Humidity','Windy','likely to rain']
print(data)
# Replace all the spaces in the column names with '_'(to replace a single string)
data.columns=data.columns.str.replace(' ','_')
print(data)
# Renaming the column names by passong them as key and value pairs in the dictionary
data.rename(columns={'Day':'day','City':'city'}, inplace=True)
print(data)
# To make changes in values of a particular row
# Here 'day' and 'likely to rain' column is changed and others are as it is
data.loc['JAI']=['Jaipur','Thursday',35,46,67,34,'YES']
print(data)
# To change the values of a particular row and column
data.loc['JAI',['day','Humidity']]=['Monday',69]
print(data)
# so apply() function works for a particular column to apply some particular function to its column values
data['city']=data['city'].apply(lambda x:x.lower())
print(data)
print(data['city'].apply(len))
# using apply() function for the dataframe
print(data.apply(len,axis='columns'))
print(data.apply(len))
# Here it applies series function
print(data.apply(lambda x:x.min()))

# so applymap() function works for the entire dataframe
print(data.applymap(lambda x: len(str(x))))

# This will map the given dataset but replace all thr rest values with NAN
# data['city']=data['city'].map({'delhi':'Delhi','jaipur':'Jaipur'})
# print(data)

# This will replace the paricular values of city without changing other values of it
data['city']=data['city'].replace({'delhi':'Delhi','jaipur':'Jaipur'})
print(data)

# To combine 2 columns and display it as a separate column in the dataframe
data['city & day']=data['city']+' '+data['day']
print(data)
# To delete the columns from the dataframe and use inplace as TRUE to make it a permanent change.
data.drop(columns=['city','day'],inplace=True)
print(data)
# To split a single column into multiple column in the the dataframe and expand is the attribute used to
# exand it in the form of table but the change is not permanent as we see, so it assigned to first and last columns.
print(data['city & day'].str.split(' ',expand=True))
print(data)
data[['city','day']]=data['city & day'].str.split(' ',expand=True)
print(data)
# Drooping the earlier combined column
data.drop(columns=['city & day'],inplace=True)
print(data)
# To add a new row in the dataframe and put ignore_index=True so that it can add the new row, as index is not
#  known prior and if any column is not specified then it takes its value as NAN.
# append method does not have inplace method so to permanently change it, we need to assign it to data
data=(data.append({'Min_temp':15,'Humidity':64,'Windy':35,
                   'likely_to_rain':'no','city':'Kanpur','day': 'Tuesday','Max_temp':38},ignore_index=True))
print(data)
# Dropping a row and using inplace method
data.drop(index=8,inplace=True)
print(data)
# Dropping a row according to a particular filter
flt=(data['day']=='Tuesday')
data.drop(index=data[flt].index,inplace=True)
print(data)

# SORTING THE DATA

# Sorting the column where default is in ascending order (if we want permanent change then set inplace=True)
print(data.sort_values(by='day'))
# Sorting by multiple columns where ascending attribute is set false for both
print(data.sort_values(by=['day','Max_temp'],ascending=False))
# Sorting by mutiple columns where day is first sorted in ascending order and within same values of day max_temp column
# is sorted in descending order.
print(data.sort_values(by=['day','Max_temp'],ascending=[True,False]))
# To get back in original form sort the index.
print(data.sort_index())

# GROUPING AND AGGREGATING