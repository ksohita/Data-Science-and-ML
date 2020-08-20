import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option('display.width',320,)
pd.set_option('display.max_columns',10)
data=pd.read_csv(r'StudentsPerformance(minor project).csv')
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

"""
So these nxt few lines gives us the count of each unique entry present in different fields of this dataset.
So from this we get to know that in our dataset,
i) We have maximum female students.
ii) Maximum students belong to Group C and minimum students belong to Group A.
iii) Most of Student's perent's have studied from some college.
iv) Most of the students take standard lunch instead of free or reduced.
v) Maximum students have not completed their test preparation course.
"""
print(data['gender'].value_counts())
print(data['race/ethnicity'].value_counts())
print(data['parental level of education'].value_counts())
print(data['lunch'].value_counts())
print(data['test preparation course'].value_counts())


sub=['math score','reading score','writing score']
# This gives us mean of scores in different subjects based on their gender.
dataset=data.groupby('gender')[sub].mean()
print(dataset)
# print(dataset.T)
xpos=np.arange((len(sub)))
# print(xpos)
y_score=np.arange(0,110,10)
male=list(dataset.T['male'])
female=list(dataset.T['female'])
#print(male)
#print(female)
w=0.25
plt.bar(xpos+w/2,male,label='Male', color='orange',width=w)
plt.bar(xpos-w/2,female,label='Female', color='green',width=w)
plt.xlabel('Subjects')
plt.ylabel('Score')
plt.title('Comparison of score of Male and Female in different subjects')
plt.legend()
plt.xticks(ticks=xpos,labels=sub)
plt.yticks(ticks=y_score,labels=y_score)
plt.tight_layout()
"""
So this bar graph shows us that male students have scored better in maths while the female students are good at 
reading and writing.
"""
plt.show()


unraces=(data['race/ethnicity'].unique())
print(unraces)
races=['group A','group B','group C','group D','group E']

xpos=np.arange(len(races))
print(xpos)
y_score=np.arange(0,110,10)

# This gives us mean of scores in different subjects based on their races or ethnicity.
dataset = data.groupby('race/ethnicity')[sub].mean()
print(dataset)
maths = list(dataset['math score'])
read = list(dataset['reading score'])
write = list(dataset['writing score'])
print(maths)
print(read)
print(write)
w=0.25
plt.bar(xpos+w,maths,label='Maths score', color='yellow',width=w)
plt.bar(xpos,read,label='Reading score', color='green',width=w)
plt.bar(xpos-w,write,label='Writing score', color='red',width=w)
plt.xlabel('race/ethnicity')
plt.ylabel('Score')
plt.title('Comparison of score of student of different races in different subjects')
plt.legend()
plt.xticks(ticks=xpos,labels=races)
plt.yticks(ticks=y_score,labels=y_score)
plt.tight_layout()
"""
So this is triple bar graph which shows us that students of 'Group E' have scored highest in all the three subjects
followed by Group D students.
"""
plt.show()


sub=['math score','reading score','writing score']

# This gives us the mean of scores in different subjects based on if they had taken their test preparation course.
dataset=data.groupby('test preparation course')[sub].mean()
print(dataset)
print(dataset.T)
xpos=np.arange((len(sub)))
print(xpos)
y_score=np.arange(0,110,10)
none=list(dataset.T['none'])
comp=list(dataset.T['completed'])
print(none)
print(comp)
w=0.25
plt.bar(xpos+w/2,none,label='None', color='pink',width=w)
plt.bar(xpos-w/2,comp,label='Completed', color='purple',width=w)
plt.xlabel('Subjects')
plt.ylabel('Score')
plt.title('Comparison of score in different subjects based on the test preparation course')
plt.legend()
plt.xticks(ticks=xpos,labels=sub)
plt.yticks(ticks=y_score,labels=y_score)
plt.tight_layout()
"""
So its a bar graph which shows us that students of who have taken the test preparation course have score higher in 
all the three subjects as compared who didn't take the course.
"""
plt.show()

uned=(data['parental level of education'].unique())
print(uned)
study = ["bachelor's degree",'some college',"master's degree","associate's degree",'high school','some high school']
xpos=np.arange(len(study))

# This gives us the mean of scores in different subjects based on their parental level of education.
dataset = data.groupby('parental level of education')[sub].mean()
print(dataset)
print(dataset.T)
maths=list(dataset['math score'])
read=list(dataset['reading score'])
write=list(dataset['writing score'])
y_score=np.arange(0,110,10)
plt.bar(xpos+w,maths,label='Maths score', color='turquoise',width=w)
plt.bar(xpos,read,label='Reading score', color='chartreuse',width=w)
plt.bar(xpos-w,write,label='Writing score', color='lavender',width=w)
plt.xlabel('parental level of education')
plt.ylabel('Score')
plt.title('Comparison of score based on their parental level of education')
plt.legend()
plt.xticks(ticks=xpos,labels=study,rotation=90)
plt.yticks(ticks=y_score,labels=y_score)
plt.tight_layout()
"""
So its a bar graph which shows us that students whose parents have Associate's Degree have scored the 
highest in all the subjects.
"""
plt.show()

# This gives us the count of students of different race/ethnicity who get free/reduced lunch and standard lunch.
dataset = data['lunch'].groupby([data['race/ethnicity'], data['lunch']]).count().unstack()
print(dataset)
free_red=list(dataset['free/reduced'])
stand=list(dataset['standard'])
y_count=np.arange(0,240,20)
xpos=np.arange(len(races))
plt.bar(xpos+w/2,free_red,label='free/reduced ', color='darkblue',width=w)
plt.bar(xpos-w/2,stand,label='standard', color='coral',width=w)
plt.xlabel('race/ethnicity')
plt.ylabel('count')
plt.title("Comparison of student's lunch based on their race/ethnicity")
plt.legend()
plt.xticks(ticks=xpos,labels=races)
plt.yticks(ticks=y_count,labels=y_count)
plt.tight_layout()
"""
So this bar graph shows us majority students all races take standard lunch while little more than half of  
students of all races take free/reduced lunch.
"""
plt.show()

"""
From this dataframe generated we get to know that parents of maximum students of Group A have degree of some high school
, parents of Group B also have degree of high school, parents of Group C have Associate's Degree, parents of Group D 
have degree of some college and parents of Group E also have Associate's Degree.

"""
dataset = data['parental level of education'].groupby([data['race/ethnicity'], data['parental level of education']]).count().unstack()
print(dataset)
# print(data['parental level of education'].unique())

datascore= data.select_dtypes(include=['int64']).copy()
print(datascore)
data['mean score']=datascore.mean(axis=1)
print(data.head())


"""
From the analysis of this data set, we get to know that the scores of the students in different subjects like Maths
Reading and Writing, mostly depends on some key features like, if they have completed their test course, their parental
level of education, the race or ethnicity they belong.
We have also compared the scores of students in different subjects according to their gender or lunch they take.
we found out the mean score of all students in these three subjects.
We also analysed the maximum parental level of education of each student of different races.
"""






