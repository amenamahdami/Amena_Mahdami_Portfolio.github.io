#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('heart.csv')
data.head(5)

sex: 1=male ; 0=female
- cp: chest pain type (4 values)
0= typical angina
1= atypical angina
2= non-anginal pain
3= asymptomatic

- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestrol in mg/dl
- fbs: fasting blood suger more than 120mg/dl (0= false; 1= true)

- restecg: resting electrochardiograpgic results
0= normal
1= having ST-T wave abnormality (Twave inversion and/or ST elevation or depression of>0.05mV)
2= showing ptobable or definite left ventrical hypertrophy by Estes' criteria

- thalach1: maximum heart rate achieved 
- exang: excerise included angina (1=yes; 0=no)

- oldpeak: ST depression induced by excersie relative to rest
- slope: the slope of the peak excerise ST segment
Value 1= upsloping
Value 2= flat
Value 3 = downsloping

- ca: number of major vessels (0-3) colored by floroscopy
- thal: 3= normal; 6=fixed defect; 7=reversable defect
- target: 0=less chance of heart attack; 1=more chance of heart
# In[3]:


data.tail(5)


# In[4]:


print('number of rows',data.shape[0])
print('number of columns',data.shape[1])


# In[5]:


data.info()


# In[6]:


print("Is there any missing value?", data.isnull().values.any())


# In[7]:


dup_data = data.duplicated().any()


# In[8]:


print("Are there any duplicated value?", dup_data)


# In[9]:


data.drop_duplicates(inplace=True)
data.info()


# In[10]:


print('number of rows',data.shape[0])


# In[11]:


data.describe(include='all')


# In[12]:


data.corr()


# In[13]:


plt.figure(figsize=(18,9))
sns.heatmap(data.corr(),annot=True)


# In[14]:


# How Many People Have Heart Disease, And How Many Don't Have Heart Disease In This Dataset?


# In[15]:


data.columns


# In[16]:


data['target'].value_counts()


# In[17]:


sns.countplot(data=data, x='target')
plt.xticks([0,1],['Less chance of Heart Disease','More chance of Heart Disease'])


# In[18]:


# Find Count of  Male & Female in this Dataset
data['sex'].value_counts()


# In[19]:


sns.countplot(data=data, x='sex')
plt.xticks([0,1],['Female','Male'])
plt.show()


# In[20]:


# Find Gender Distribution According to The Target Variable
sns.countplot(data=data, x='sex', hue='target')
plt.xticks([0,1],['Female','Male'])
plt.legend(labels=['No Disease','Disease'])
plt.xlabel("Gender",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[21]:


# Check Age Distribution In The Dataset
sns.displot(data['age'],bins=20)
plt.show()


# In[ ]:





# # Check Chest Pain Type
# - cp: chest pain type (4 values)
# 0= typical angina
# 1= atypical angina
# 2= non-anginal pain
# 30 asymptomatic

# In[22]:


sns.countplot(data=data, x='cp')
plt.xticks([0,1,2,3],['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],rotation=30)
plt.xlabel("Chest Pain Type",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# In[23]:


# Show The Chest Pain Distribution As Per Target Variable

sns.countplot(data=data, x='cp', hue='target')
plt.xticks([0,1,2,3],['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],rotation=30)
plt.xlabel("Chest Pain Type",fontsize=14)
plt.legend(labels=['No Disease','Disease'])
plt.ylabel("Count",fontsize=14)
plt.show()


# In[24]:


# Show Fasting Blood Sugar Distribution According To Target Variable
sns.countplot(data=data, x='fbs', hue='target')
plt.xticks([0,1],['False','True'])
plt.xlabel("Fasting Blood Suger More Than 120mg/dl",fontsize=14)
plt.legend(labels=['No Disease','Disease'])
plt.ylabel("Count",fontsize=14)
plt.show()


# In[25]:


# Check Resting Blood Pressure Distribution
data['trestbps'].hist()


# In[26]:


# Compare Resting Blood Pressure As Per Sex Column
g = sns.FacetGrid(data,hue='sex',aspect=3)
g.map(sns.kdeplot,'trestbps',fill=True)
plt.legend(labels=['Male','Female'])
plt.xlabel("Resting Blood Pressure",fontsize=14)


# In[27]:


# Show Distribution of Serum cholesterol
data['chol'].hist()


# In[28]:


# Plot Continuous Variables
data.columns


# Spliting Categorical columns from Continuous columns is an important part of data processing

# In[29]:


#first we split columns which have Continuous values and columns with categorical values
cat_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique(0) <=10:
        cat_val.append(column)
    else:
        cont_val.append(column)


# In[30]:


cat_val


# In[31]:


cont_val


# In[32]:


data.hist(cont_val, figsize=(15,15))
plt.tight_layout()
plt.show()
# we can notice here the oldpeak is left skewed and the thalach is right skewed


# ## Prediction of Heart Disease incident using Machine Learning

# 1. Econding of Categorical data
# 
# Here the numbers are encoding a pain type, but ML model does not understand that
# 
# So we need to convert each value into one column with binary values; four columns in this case
# 
# Also called as "Dummy Variables"
# 
# The Dummy Trap: a senario in which the independent variable are highly correlated, means; one variable canbe predicted from other variables

# In[33]:


cat_val


# In[34]:


data['cp'].unique()


# In[35]:


#we will remove 'sex' and 'target' columns from the list bc they already contain binary values in their column
cat_val.remove('sex')
cat_val.remove('target')


# In[36]:


data=pd.get_dummies(data,columns=cat_val, dtype=int,drop_first=True)


# In[37]:


data.head(2)


# ## Feature Scaling
# 
# FS allows us to put our continous variables (features) into the same scale
# 
# It's essintial for ML algorithm that calculate distance bw data
# 
# If not scaled, the features with highest valeus start dominating distances
# 
# Any ML algorith which is not distance-based, is not affected by FS

# In[38]:


from sklearn.preprocessing import StandardScaler


# In[39]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[40]:


data.head(5)


# ## Splitting The Dataset Into The Training Set And Test Set
# 
# 1-first split independent variables from dependent variable

# In[42]:


X = data.drop('target',axis=1)


# In[43]:


y = data['target']


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[53]:


X_train


# In[54]:


X_test


# In[55]:


y_train


# In[56]:


y_test


# ## Logistic Regression
# 
# - our 'target' column has values 0 and 1, a categorical variable ; so this is a classification problem
# 
# - Using diffrent classification algorithms, and evaluate their performace based on accurecy, and will find the best model for this particular dataset
# 
# 1- We start by training our model

# In[57]:


data.head(5)


# In[58]:


from sklearn.linear_model import LogisticRegression


# In[60]:


# create instacne for this logistic regression 

log = LogisticRegression()
#train the model on our training set
log.fit(X_train,y_train)


# In[61]:


# Now our model is trained, now we will perform predication:
y_pred1 = log.predict(X_test)


# In[62]:


# Now let's check how accurate this model is:
from sklearn.metrics import accuracy_score


# In[63]:


accuracy_score(y_test,y_pred1)


# Our Logistic Regression model is around 79% accurate for this particular dataset

# ## SVC model
# 
# Support Vector Classifier

# In[64]:


from sklearn import svm


# In[65]:


# create instacne for this model
svm = svm.SVC()


# In[66]:


# train the model on our training set
svm.fit(X_train,y_train)


# In[67]:


# perform prediction using samples that we have saved in X_test
y_pred2 = svm.predict(X_test)


# In[68]:


# Check the SVC model for accuracy
accuracy_score(y_test,y_pred2)


# Our Support Vector Classifier model is around 80% accurate for this particular dataset

# ## K nearest Neighbors Classifier
# 

# In[69]:


from sklearn.neighbors import KNeighborsClassifier


# In[71]:


# create instacne for this model
knn = KNeighborsClassifier()


# In[72]:


# train the model on our training set
knn.fit(X_train,y_train)


# In[76]:


# perform prediction using samples that we have saved in X_test
y_pred3=knn.predict(X_test)


# In[74]:


# Check the KNN model for accuracy
accuracy_score(y_test,y_pred3)


# In[78]:


score = []
for k in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))


# In[79]:


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)


# ## Non-Linear ML Algorithms
# 
# for non-linear ML algorithm, it's not required to perform any intiger scaling
# 
# 12. Decision Tree Classifier
# 13. Random Forest Classifier
# 14. Gradient Boosting Classifier
# 15. Prediction on New Data
# 16. Save Model Usign Joblib
# 17. Creating GUI

# In[80]:


#let's upload the data once again:
data = pd.read_csv('heart.csv')


# In[81]:


data = data.drop_duplicates()


# In[82]:


X = data.drop('target',axis=1)
y = data['target']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# ## Decision Tree Classifier
# 

# In[83]:


from sklearn.tree import DecisionTreeClassifier


# In[84]:


dt = DecisionTreeClassifier()


# In[86]:


dt.fit(X_train,y_train)


# In[115]:


y_pred4 = dt.predict(X_test)


# In[88]:


accuracy_score(y_test,y_pred4)


# Our decision tree classifier is 72% accurate

# ## Random Forest Classifier 

# In[97]:


from sklearn.ensemble import RandomForestClassifier


# In[124]:


rf = RandomForestClassifier()


# In[128]:


rf.fit(X_train,y_train)


# In[129]:


y_pred5 = rf.predict(X_test)


# In[130]:


accuracy_score(y_test,y_pred5)


# Our Random forest classifier is 80% accurate

# ## Gradient Boosting Classifier

# In[131]:


from sklearn.ensemble import GradientBoostingClassifier


# In[132]:


gbc = GradientBoostingClassifier()


# In[133]:


gbc.fit(X_train,y_train)


# In[134]:


y_pred6 = rf.predict(X_test)


# In[135]:


accuracy_score(y_test,y_pred6)


# Our Gradient Boosting Classifier is 80% accurate

# In[137]:


final_data = pd.DataFrame({'Models':['LR','SVM','DT','RF','GB'],
                           'ACC':[accuracy_score(y_test,y_pred1),
                                 accuracy_score(y_test,y_pred2),
                                 accuracy_score(y_test,y_pred4),
                                 accuracy_score(y_test,y_pred5),
                                 accuracy_score(y_test,y_pred6)]})


# In[138]:


final_data


# In[141]:


sns.barplot(data=final_data,x='Models',y='ACC')


# We can choose the best model for our production
# 
# we did the train_test_split just to evaluate the performace of our models
# 
# for production, we have to train our model on entire dataset
# 
# So we will train our best model on entire dataset
# 

# In[142]:


X = data.drop('target',axis=1)
y = data['target']


# In[143]:


X.shape


# In[144]:


from sklearn.ensemble import RandomForestClassifier


# In[145]:


#train RFC on entire dataset
rf = RandomForestClassifier()
rf.fit(X,y)


# In[147]:


data.columns


# ## Prediction on New Data

# In[148]:


new_data = pd.DataFrame({'age':52, 'sex':1, 'cp':0, 'trestbps':125, 'chol':212, 'fbs':0, 'restecg':1, 'thalach':168,
       'exang':0, 'oldpeak':1.0, 'slope':2, 'ca':2, 'thal':3,
                        },index=[0])


# In[149]:


new_data
#we will perfrom prediction on this patient's data to find out if they will have heart diseae or not


# In[150]:


p = rf.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")
    


# # Save the ML model using joblib

# In[151]:


import joblib 


# In[152]:


# right in the ( the instace of the job model)
joblib.dump(rf,'Model_jobleb_heart')


# In[157]:


# you have to load this model
model = joblib.load('Model_jobleb_heart')


# In[158]:


model.predict(new_data)


# # develop GUI for our ML model

# In[161]:


from tkinter import *
import joblib


# In[162]:


from tkinter import *
import joblib
import numpy as np
from sklearn import *
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Heart Disease").grid(row=31)
    else:
        Label(master, text="Possibility of Heart Disease").grid(row=31)
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white"). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="Enter Value of CP").grid(row=3)
Label(master, text="Enter Value of trestbps").grid(row=4)
Label(master, text="Enter Value of chol").grid(row=5)
Label(master, text="Enter Value of fbs").grid(row=6)
Label(master, text="Enter Value of restecg").grid(row=7)
Label(master, text="Enter Value of thalach").grid(row=8)
Label(master, text="Enter Value of exang").grid(row=9)
Label(master, text="Enter Value of oldpeak").grid(row=10)
Label(master, text="Enter Value of slope").grid(row=11)
Label(master, text="Enter Value of ca").grid(row=12)
Label(master, text="Enter Value of thal").grid(row=13)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)


Button(master, text='Predict', command=show_entry_fields).grid()

mainloop()


# In[ ]:




