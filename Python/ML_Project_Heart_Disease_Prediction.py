#!/usr/bin/env python
# coding: utf-8

# # 1- Dataset Wrangling and Exploration:

# ### Import Libraries and Dataset

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('heart.csv')


# ### Display dataset 

# In[5]:


data.head(5)


# ### Variables description:
# 
# - sex: 1=male ; 0=female
# 
# - cp: chest pain type (4 values)
# 0= typical angina
# 1= atypical angina
# 2= non-anginal pain
# 3= asymptomatic
# 
# - trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# - chol: serum cholestrol in mg/dl
# - fbs: fasting blood suger more than 120mg/dl (0= false; 1= true)
# 
# - restecg: resting electrochardiograpgic results
# 0= normal
# 1= having ST-T wave abnormality (Twave inversion and/or ST elevation or depression of>0.05mV)
# 2= showing ptobable or definite left ventrical hypertrophy by Estes' criteria
# 
# - thalach1: maximum heart rate achieved 
# - exang: excerise included angina (1=yes; 0=no)
# 
# - oldpeak: ST depression induced by excersie relative to rest
# - slope: the slope of the peak excerise ST segment
# Value 1= upsloping
# Value 2= flat
# Value 3 = downsloping
# 
# - ca: number of major vessels (0-3) colored by floroscopy
# - thal: 3= normal; 6=fixed defect; 7=reversable defect
# - target: 0=less chance of heart attack; 1=more chance of heart

# In[6]:


data.tail(5)


# In[7]:


print('number of rows',data.shape[0])
print('number of columns',data.shape[1])


# In[8]:


data.info()


# ### Check for null values or duplicates in the dataset

# In[9]:


print("Is there any missing value?", data.isnull().values.any())


# In[10]:


dup_data = data.duplicated().any()


# In[11]:


print("Are there any duplicated value?", dup_data)


# In[12]:


data.drop_duplicates(inplace=True)
data.info()


# In[14]:


print('number of rows',data.shape[0])


# ### Get overall statistics about the dataset

# In[15]:


data.describe(include='all')


# ### Draw Correlation Matrix

# In[16]:


data.corr()


# In[17]:


plt.figure(figsize=(18,9))
sns.heatmap(data.corr(),annot=True)


# ## Data exploration

# ### How many people have heart disease, and how many don't have heart disease in this dataset?

# In[19]:


data.columns


# In[20]:


data['target'].value_counts()


# In[21]:


sns.countplot(data=data, x='target')
plt.xticks([0,1],['Less chance of Heart Disease','More chance of Heart Disease'])


# ### Find count of  Male & Female in this dataset

# In[22]:


data['sex'].value_counts()


# In[23]:


sns.countplot(data=data, x='sex')
plt.xticks([0,1],['Female','Male'])
plt.show()


# ### Find Gender distribution according to the Target variable

# In[24]:


sns.countplot(data=data, x='sex', hue='target')
plt.xticks([0,1],['Female','Male'])
plt.legend(labels=['No Disease','Disease'])
plt.xlabel("Gender",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# ### Check Age distribution

# In[25]:


sns.displot(data['age'],bins=20)
plt.show()


# ### Check Chest Pain types
# - cp: chest pain type (4 values)
# 0= typical angina
# 1= atypical angina
# 2= non-anginal pain
# 30 asymptomatic

# In[26]:


sns.countplot(data=data, x='cp')
plt.xticks([0,1,2,3],['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],rotation=30)
plt.xlabel("Chest Pain Type",fontsize=14)
plt.ylabel("Count",fontsize=14)
plt.show()


# ### Show the Chest Pain distribution as per Target variable

# In[27]:


sns.countplot(data=data, x='cp', hue='target')
plt.xticks([0,1,2,3],['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'],rotation=30)
plt.xlabel("Chest Pain Type",fontsize=14)
plt.legend(labels=['No Disease','Disease'])
plt.ylabel("Count",fontsize=14)
plt.show()


# ### Show Fasting Blood Sugar distribution according to Target variable

# In[29]:


sns.countplot(data=data, x='fbs', hue='target')
plt.xticks([0,1],['False','True'])
plt.xlabel("Fasting Blood Suger More Than 120mg/dl",fontsize=14)
plt.legend(labels=['No Disease','Disease'])
plt.ylabel("Count",fontsize=14)
plt.show()


# ### Check Resting Blood Pressure distribution

# In[25]:


data['trestbps'].hist()


# ### Compare Resting Blood Pressure as per Gender

# In[32]:


g = sns.FacetGrid(data,hue='sex',aspect=3)
g.map(sns.kdeplot,'trestbps',fill=True)
plt.legend(labels=['Male','Female'])
plt.xlabel("Resting Blood Pressure",fontsize=14)


# ### Show distribution of Serum Cholesterol

# In[33]:


data['chol'].hist()


# ### Plot Continuous Variables

# In[28]:


data.columns


# ### Split columns which have Continuous values and columns with categorical values
# 
# (Spliting Categorical columns from Continuous columns is an important part of data processing)

# In[34]:


cat_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique(0) <=10:
        cat_val.append(column)
    else:
        cont_val.append(column)


# In[35]:


cat_val


# In[36]:


cont_val


# In[32]:


data.hist(cont_val, figsize=(15,15))
plt.tight_layout()
plt.show()


# We can notice here the oldpeak is left skewed and the thalach is right skewed

# # 2- Prediction of Heart Disease incident using Machine Learning:

# ### Econding of Categorical data
# 
# Here the numbers are encoding a pain type, but ML model does not understand that
# 
# So we need to convert each value into one column with binary values; four columns in this case
# 
# Also called as "Dummy Variables"
# 
# The Dummy Trap: a senario in which the independent variable are highly correlated, means; one variable can be predicted from other variables

# In[37]:


cat_val


# In[38]:


data['cp'].unique()


# Remove 'sex' and 'target' columns from the list bc they already contain binary values in their column

# In[39]:


cat_val.remove('sex')
cat_val.remove('target')


# In[40]:


data=pd.get_dummies(data,columns=cat_val, dtype=int,drop_first=True)


# In[41]:


data.head(2)


# ### Feature Scaling
# 
# FS allows us to put our continous variables (features) into the same scale
# 
# It's essintial for ML algorithm that calculate distance between data
# 
# If not scaled, the features with highest valeus start dominating distances
# 
# Any ML algorith which is not distance-based, is not affected by FS

# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])


# In[44]:


data.head(5)


# ### Splitting the dataset into the Training set and Test set
# 
# First split independent variables from dependent variable

# In[45]:


X = data.drop('target',axis=1)


# In[46]:


y = data['target']


# Then perform train test split

# In[51]:


from sklearn.model_selection import train_test_split


# In[50]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[52]:


X_train


# In[53]:


X_test


# In[54]:


y_train


# In[56]:


y_test


# 
# 
# Our 'target' column has values 0 and 1, a categorical variable; so this is a classification problem
# 
# Steps:
# - Use different classification algorithms
# - Evaluate their performace based on accurecy
# - Find the best model for this particular dataset

# In[57]:


data.head(5)


# ### Logistic Regression

# In[59]:


from sklearn.linear_model import LogisticRegression


# Create instacne for this logistic regression:

# In[60]:


log = LogisticRegression()


# Train the model on our training set:

# In[61]:


log.fit(X_train,y_train)


# Our model is trained, now we will perform predication:

# In[62]:


y_pred1 = log.predict(X_test)


# Let's check how accurate this model is:

# In[63]:


from sklearn.metrics import accuracy_score


# In[64]:


accuracy_score(y_test,y_pred1)


# Our Logistic Regression model is around 79% accurate for this particular dataset

# ### SVC 
# 
# Support Vector Classifier

# In[65]:


from sklearn import svm


# Create instacne for this model

# In[66]:


svm = svm.SVC()


# Train the model on our training set

# In[67]:


svm.fit(X_train,y_train)


# Perform prediction using samples that we have saved in X_test

# In[69]:


y_pred2 = svm.predict(X_test)


# Check the SVC model for accuracy

# In[70]:


accuracy_score(y_test,y_pred2)


# Our Support Vector Classifier model is around 80% accurate for this particular dataset

# ### Non-Linear ML Algorithms
# 
# For non-linear ML algorithm, it's not required to perform any intiger scaling
# 
# - Decision Tree Classifier
# - Random Forest Classifier
# - Gradient Boosting Classifier
# - Prediction on New Data
# - Save Model Usign Joblib
# - Creating GUI

# Let's upload the data once again:

# In[80]:


data = pd.read_csv('heart.csv')


# In[81]:


data = data.drop_duplicates()


# In[82]:


X = data.drop('target',axis=1)
y = data['target']


# In[83]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# ### Decision Tree Classifier
# 

# In[84]:


from sklearn.tree import DecisionTreeClassifier


# In[85]:


dt = DecisionTreeClassifier()


# In[86]:


dt.fit(X_train,y_train)


# In[87]:


y_pred4 = dt.predict(X_test)


# In[88]:


accuracy_score(y_test,y_pred4)


# Our Decision Tree Classifier is 72% accurate

# ## Random Forest Classifier 

# In[89]:


from sklearn.ensemble import RandomForestClassifier


# In[90]:


rf = RandomForestClassifier()


# In[91]:


rf.fit(X_train,y_train)


# In[92]:


y_pred5 = rf.predict(X_test)


# In[93]:


accuracy_score(y_test,y_pred5)


# Our Random forest classifier is 80% accurate

# ## Gradient Boosting Classifier

# In[94]:


from sklearn.ensemble import GradientBoostingClassifier


# In[95]:


gbc = GradientBoostingClassifier()


# In[96]:


gbc.fit(X_train,y_train)


# In[97]:


y_pred6 = rf.predict(X_test)


# In[98]:


accuracy_score(y_test,y_pred6)


# Our Gradient Boosting Classifier is 80% accurate

# ### Compare different accuracy scores provided by different models:

# In[103]:


final_data = pd.DataFrame({'Models':['LR','SVC','DTC','RFC','GBC'],
                           'ACC':[accuracy_score(y_test,y_pred1),
                                 accuracy_score(y_test,y_pred2),
                                 accuracy_score(y_test,y_pred4),
                                 accuracy_score(y_test,y_pred5),
                                 accuracy_score(y_test,y_pred6)]})


# In[104]:


final_data


# In[105]:


sns.barplot(data=final_data,x='Models',y='ACC')


# We can choose the best model for our production, Random Forest Classifier and Gradient Boosting Classifier had equally the highest sccuracy score
# 
# Remember that we did the train_test_split just to evaluate the performace of our models. for production, we have to train our model on entire dataset
# 
# So we will train our best model (I will choose Random Forest Classifier) on the entire dataset
# 

# In[108]:


X = data.drop('target',axis=1)
y = data['target']


# In[109]:


X.shape


# In[110]:


from sklearn.ensemble import RandomForestClassifier


# Train RFC on entire dataset

# In[112]:


rf = RandomForestClassifier()
rf.fit(X,y)


# In[113]:


data.columns


# ### Prediction on new data
# 
# We will perfrom prediction on this patient's data to find out if they will have heart diseae or not

# In[116]:


new_data = pd.DataFrame({'age':52, 'sex':1, 'cp':0, 'trestbps':125, 'chol':212, 'fbs':0, 'restecg':1, 'thalach':168,
       'exang':0, 'oldpeak':1.0, 'slope':2, 'ca':2, 'thal':3,
                        },index=[0])


# In[117]:


new_data


# In[118]:


p = rf.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")
    


# ### Save the ML model using joblib

# In[119]:


import joblib 


# In[121]:


joblib.dump(rf,'Model_jobleb_heart')


# Load this model

# In[122]:


model = joblib.load('Model_jobleb_heart')


# In[123]:


model.predict(new_data)


# 0=No Disease,
# 1=Disease
