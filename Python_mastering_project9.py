#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/PromptCloudHQ/imdb-data

# ### IMDB data from 2006 to 2016
Here's a data set of 1,000 most popular movies on IMDB in the last 10 years. The data points included are:

Title, Genre, Description, Director, Actors, Year, Runtime, Rating, Votes, Revenue, Metascrore
# In[99]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


data=pd.read_csv('IMDB-Movie-Data.csv')


# ### 1. Display Top 10 Rows of The Dataset

# In[65]:


data.head(10)


# ### 2. Check Last 10 Rows of The Dataset

# In[67]:


data.tail(10)


# ### 3. Find Shape of Our Dataset (Number of Rows And Number of Columns)

# In[6]:


print('number of rows',data.shape[0])
print('number of columns',data.shape[1])


# ### 4. Getting Information About Our Dataset Like Total Number Rows, Total Number of Columns, Datatypes of Each Column And Memory Requirement

# In[7]:


data.info()


# ### 5. Check Missing Values In The Dataset

# In[6]:


print("Is there any missing value?", data.isnull().values.any())


# In[7]:


data.isnull().sum()


# In[14]:


sns.heatmap(data.isnull())


# In[15]:


data.isnull().sum()*100/len(data)


# ### 6. Drop All The  Missing Values

# In[17]:


data.dropna(axis=0,inplace=True)
data


# In[8]:





# ### 7. Check For Duplicate Data

# In[69]:


dup_data=data.duplicated().any()


# In[70]:


print('Are there any duplicate value?', dup_data)


# ### 8. Get Overall Statistics About The DataFrame

# In[71]:


data.describe(include='all')


# ### 9. Display Title of The Movie Having Runtime >= 180 Minutes

# In[73]:


data.columns


# In[72]:


data[data['Runtime (Minutes)']>=180]['Title']


# ### 10. In Which Year There Was The Highest Average Voting?

# In[74]:


data.groupby(['Year'])['Votes'].mean().sort_values(ascending=False)


# In[11]:


sns.barplot(x='Year',y='Votes',data=data)
plt.title("Votes by Year")
plt.show()


# ### 11. In Which Year There Was The Highest Average Revenue?

# In[44]:


data.groupby(['Year'])['Revenue (Millions)'].mean().sort_values(ascending=False)


# In[12]:


sns.barplot(x='Year',y='Revenue (Millions)',data=data)
plt.title("Revenue by Year")
plt.show()


# ### 12. Find The Average Rating For Each Director

# In[25]:


data.columns


# In[47]:


data.groupby(['Director'])['Rating'].mean().sort_values(ascending=False)


# ### 13. Display Top 10 Lengthy Movies Title and Runtime

# In[75]:


top10_len=data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']]\
.set_index('Title')


# In[76]:


top10_len


# In[77]:


sns.barplot(x='Runtime (Minutes)', y=top10_len.index, data=top10_len)
plt.show()


# ### 14. Display Number of Movies Per Year

# In[78]:


data['Year'].value_counts(ascending=False)


# In[18]:


sns.countplot(x='Year',data=data)
plt.title("Number of Movies Per Year")
plt.show()


# ### 15. Find Most Popular Movie Title (Higest Revenue)

# In[34]:


data.columns


# In[90]:


data[data['Revenue (Millions)'].max()==data['Revenue (Millions)']]['Title']


# ### 16. Display Top 10 Highest Rated Movie Titles And its Directors

# In[22]:


top10=data.sort_values(by='Rating',ascending=False).head(10)\
.set_index('Title')
top10


# In[25]:


top10[['Rating','Director']]


# In[32]:


sns.barplot(x='Rating', y=top10.index, data=top10,hue='Director',dodge=False)
plt.legend(bbox_to_anchor=(1.05,1),loc=2)
plt.title("Top 10 Rating Movies and Their Directors")
plt.show()


# ### 17. Display Top 10 Highest Revenue Movie Titles

# In[81]:


top10Rev=data.sort_values(by='Revenue (Millions)',ascending=False).head(10)
top10Rev['Title']


# In[86]:


top10Rev[['Title','Revenue (Millions)']]


# In[82]:


#OR
data.nlargest(10,'Revenue (Millions)')


# In[83]:


Top10R=data.nlargest(10,'Revenue (Millions)')[['Title','Revenue (Millions)']]\
.set_index('Title')


# In[84]:


Top10R


# In[87]:


sns.barplot(x='Revenue (Millions)', y='Title', data=top10Rev)


# In[88]:


#or
sns.barplot(x='Revenue (Millions)', y=Top10R.index, data=Top10R)
plt.title("Top 10 Highest Revenue Movie Titles")
plt.show()


# ### 18.  Find Average Rating of Movies Year Wise

# In[89]:


data.groupby('Year')['Rating'].mean()


# ### 19. Does Rating Affect The Revenue?

# In[90]:


plt.figure(figsize=(15,6))
sns.scatterplot(x='Rating', y='Revenue (Millions)', data=data)


# ### 20. Classify Movies Based on Ratings [Excellent, Good and Average]

# In[91]:


def categorize_rating(rating):
    if rating >= 7.0:
        return 'Excellent'
    elif rating >= 5.0:
        return 'Good'
    else:
        return 'Average'


# In[92]:


data['rating_cat'] = data['Rating'].apply(categorize_rating)


# In[94]:


data.head(5)


# ### 21. Count Number of Action Movies

# In[48]:


data.columns


# In[49]:


data['Genre'].dtype


# In[95]:


len(data[data['Genre'].str.contains('Action',case=False)])


# ### 22. Find Unique Values From Genre 

# In[52]:


data['Genre']


# ### Step 1: create a list with entire values of Genre column, split by ,

# In[53]:


list1=[]
for value in data['Genre']:
    list1.append(value.split(','))


# In[96]:


list1


# #### Step 2 convert this lenghty list into a shorter list with 1 value per coma
# 

# In[97]:


one_d=[]
for item in list1:
    for item1 in item:
        one_d.append(item1)
    


# In[98]:


one_d


# #### Step 3 find out unique values in the previuous lidt with one value
# 

# In[57]:


uni_list=[]
for item in one_d:
    if item not in uni_list:
        uni_list.append(item)
        


# In[58]:


uni_list


# ### 23. How Many Films of Each Genre Were Made?

# In[65]:


one_d


# #### import collections module to use Counter ()
# 

# In[61]:


from collections import Counter


# In[64]:


Counter(one_d)


# ### Other usful functions not use in the dataset above:

# In[ ]:


#shift+tab to see the arguments


# In[ ]:


# to display values in decimal format instead of exponentioal format to read it properly:
pd.options.display.float_format='{:.2f}'.format


# In[ ]:


# Data Cleaning:  (Replace '--'  to NaN)
data=data.replace('--',np.nan,regex=True)
data['CoulumnName']=data['CoulumnName'].replace('--',np.nan)

# Drop all The Missing Values
data.dropna(how='any',inplace=True)
# Fill missing values with 0
data['CoulumnName'] = data['CoulumnName'].fillna(0)


# In[ ]:


# Drop/remove columns
data = data.drop(['CoulumnName1','CoulumnName2','CoulumnName3'],axis=1)
#or
data.drop(['CoulumnName1','CoulumnName2','CoulumnName3'],axis=1,inplace=True)


# In[ ]:


# To remove 2 string at the end of each value in a column using slicing
data['CoulumnName']=data['CoulumnName'].str[0:-2]


# In[ ]:


# Fetch Random Sample From the Dataset (50%)
#frac from fraction
data.sample(frac=0.50)
# Everytime we run the sample() we get a diffrent sample , 
#to fix it we use random_state=100
data.sample(frac=0.50,random_state=100)


# In[ ]:


# What are the top 5 most popular email providers? method using apply(lambda x:x)
data['Email'].apply(lambda x:x.split('@')[1]).value_counts().head(5)
# How many people have a credit card that expires in 2020?
data[data['CC Exp Date'].apply(lambda x:x[3:]=='20')].count()


# In[ ]:


# To remove the comas from numbers in a column & convert dtype to int64
data['CoulumnName']=data['CoulumnName'].str.replace(',','').astype('int')


# In[ ]:


# To convert numeric column dtype to int
data['CoulumnName']=data['CoulumnName'].astype('int')
# to conver XColumn Datatype To Category Datatype
data['CoulumnName'] = data['CoulumnName'].astype('category')
data['CoulumnName'] = data['CoulumnName'].astype('str')

#you can see that you can optimize the memory usage by changing the data type.


# In[ ]:


# To map the string values in a column into numerical values
data['CoulumnName']=data['CoulumnName'].map({'A++ ':5,'A+ ':4,'A ':3,'A- ':2,'B+ ':1})


# In[ ]:


# to upload a dataset that contains a column with timestamp
#data = pd.read_csv('DatasetName.csv',parse_dates=['CoulumnName_with_timestamp'])


# In[ ]:


# To display certain columns alone at the same time
data[['CoulumnName1', 'CoulumnName2', 'CoulumnName3']]


# In[ ]:


# filter: select rows as per condition(s)
sum(data['CoulumnName']=='value_in_column')
# to view dataframe
data[data['CoulumnName']=='value_in_column'].head(5)


# In[ ]:


# Handeling missing values if not too many: String data
# mode() the mode is the value that appears most often in a set of data values in a coumn
# means the most frequent occuring values in the column
data['CoulumnName'].mode()
#fill in missing values in the column with the most common value
data['CoulumnName'].fillna('common_value',inplace=True)


# In[ ]:


# Handling Missing Values: numberical data type
# fill in the missing values with the mean() value of the numerical column
data['Numerical_CoulumnName'].fillna(data['Numerical_CoulumnName'].mean(),inplace=True)


# In[ ]:


# Categorical Data Encoding
# why it's importatnt to do encoding? 
#bc in the future we have to provide these values to some machine learning algorithm
#Which are mathematical algorithims. they can't understand string values

#Important to remember before encoding to know how many categories are there in the column
data['Cat_CoulumnName'].unique()
data['Cat_CoulumnName'].map({'ABC':1,'DEF':0})

# to modify our existing dataframe: adding a new column, which by default is added at the end of the df
data['New_CoulumnName']=data['Cat_CoulumnName'].map({'ABC':1,'DEF':0})
#to change the position of a new column, here is how you create it:
# x is the 'value' needed for insert()
x=data['Cat_CoulumnName'].map({'ABC':1,'DEF':0})
data.insert(5,'New_CoulumnName',x)
#has to be a new column name


# In[ ]:


# What if we have large number of categories? it's better to use dumies method of panda 
data['Cat_CoulumnName'].nunique()
data['Cat_CoulumnName'].unique()
pd.get_dummies(data,columns=['Cat_CoulumnName'], dtype=float)
# you can see here that the 'Cat_CoulumnName' was deleted and 3 columns created instead
#each column represent 1 value of the 'Cat_CoulumnName'
#by knowing the value (0 or 1) of 2 columns, you can predict the value of the third column
#So you can drop one column, knowing that you can predict its value from the other 2 columns
data1=pd.get_dummies(data,columns=['Cat_CoulumnName'], dtype=float, drop_first=True)
data1.head(1)
# dtype=float argument will convert it to 0/1 instead of false/true


# In[ ]:


### Univariate Analysis
# meaning we are taking one variable at a time and performing analysis on it
# We are intersted in 1 variable only
# It does not analyze any relationships
# the purpose of it is to describe the data and find patterns exists within it


# In[ ]:


data['Survived'].value_counts()


# In[ ]:


# count plot : to find out the counts in an object or categorical column
sns.countplot(data=data, x='Cat_CoulumnName')
plt.xlabel('Cat_CoulumnName', fontsize=13)
plt.ylabel('Countof_X')
plt.xticks(rotation=60)
plt.show()


# In[ ]:


# we can find the Distribution by histogram:
plt.figure(figsize=(10,10))
data['CoulumnName'].hist()


# In[ ]:


# Using between method:
sum(data['Numerical_CoulumnName'].between(17,48))
# here it will find Total Number of counts Having value Between 17 To 48 


# In[ ]:


# to find out counts having certain conditions in a column
data['CoulumnName'].unique()
filter1=data['CoulumnName']=='condition1'
filter2=data['CoulumnName']=='condition2'
len(data[filter1 | filter2])
## OR
sum(data['education'].isin(['Bachelors','Masters']))


# In[ ]:


### Bivariate Analysis
# it's used to find relationships between two diffrent variables
# something as simple as createing scatter plot or box plot


# In[ ]:


# barplot
# Please remember that barplots are useful to visualize the relationship bw categorical data and atleast on numerical variable
sns.barplot(x='Cat_CoulumnName', y='Numerical_CoulumnName', data=data)
#Barplots are also used when comparing categoris of data, in two categorical variables


# In[ ]:


# scatterplot 
#to find the relationship bw two numeric columns
plt.figure(figsize=(15,6))
sns.scatterplot(x='Numerical_CoulumnName', y='Numerical_CoulumnName', data=data)


# In[ ]:


# To Find Correlation Matrix in a dataset, that has all int or float columns, excep for one oject column
correlation_matrix = data.drop(columns='ObjectCoulumnName').corr()
correlation_matrix


# In[ ]:


#### Feature Engineering
# It's the process of using domain knowlege to extract features from raw data, via data mining techniques
# Then these features can be used to improve machine learning algorithms


# In[ ]:


# To create new columns with help of existing columns
# 'SibSp', 'Parch' with help of these two columns we can find the size of persons family
data['new_CoulumnName']=data['CoulumnName1']+data['CoulumnName3']
data['new_CoulumnName']=data['CoulumnName1']/data['CoulumnName3']


# In[ ]:




