#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import plotly
import plotly.express as px

import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
from sklearn.metrics import accuracy_score,mean_squared_error

import cufflinks as cf
pyo.init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


df=pd.read_csv("heart.csv")


# In[4]:


df


# In[5]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])


# In[6]:


df['target']


# In[7]:


df.groupby('target').size()


# In[8]:


df.shape


# In[9]:


df.size


# In[10]:


df.describe


# In[11]:


df.describe()


# In[12]:


df.info()


# In[13]:


#letusesomeVisualization


# In[14]:


df.hist(figsize=(15,15)) #pandasvisualization but runs on top of matplotlib
plt.show()


# In[15]:


sns.barplot(df['sex'],df['target'])
plt.show()


# In[16]:


sns.barplot(df['sex'],df['age'],hue=df['target'])
plt.show()


# In[ ]:





# In[17]:


sns.distplot(df["chol"])
plt.show()


# In[18]:


y = df["target"]

sns.countplot(y)

target_temp = df.target.value_counts()

print(target_temp)


# In[19]:


numeric_columns=['age','trestbps','chol','oldpeak','thalach']


# In[20]:


sns.heatmap(df[numeric_columns].corr(),annot=True,cmap='terrain',linewidths=0.1)


# In[21]:


# create four distplots
plt.figure(figsize=(14,12))
plt.subplot(221)
sns.distplot(df[df['target']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(df[df['target']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(df[df['target']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(df[df['target']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()


# In[22]:


#datapreprocessing


# In[23]:


x,y=df.loc[:,:'thal'],df['target']


# In[24]:


x


# In[25]:


y


# In[26]:


df.shape


# In[27]:


x.shape


# In[28]:


y.shape


# In[29]:


y.size


# In[30]:


x.size


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)


# In[33]:


x_train


# In[34]:


x_train.shape


# In[35]:


y_train


# In[36]:


y_train.shape


# In[37]:


x_test


# In[38]:


y_test


# In[39]:


x_test.shape


# In[40]:


y_test.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


x_train


# In[42]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[43]:


x_test


# In[ ]:





# In[44]:


x_test


# In[ ]:





# In[45]:


prediction=dt.predict(x_test)


# In[46]:


prediction


# In[47]:


x_test


# In[48]:


accuracy_dt=accuracy_score(y_test,prediction)*100


# In[49]:


accuracy_dt


# In[50]:


dt.feature_importances_


# In[51]:


def plot_feature_importances(model):
    plt.figure(figsize=(8,6))
    n_features = 13
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


# In[52]:


plot_feature_importances(dt)


# In[53]:


Category=['No,You donot have Heart disease','Sorry ,You are having heart disease']


# In[54]:


custom_data=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
custom_data_prediction_dt=dt.predict(custom_data)


# In[55]:


custom_data_prediction_dt


# In[56]:


print(Category[int(custom_data_prediction_dt)])


# In[57]:


#knn


# In[ ]:





# In[ ]:





# In[58]:


from sklearn.neighbors import KNeighborsClassifier


# In[59]:


from sklearn.preprocessing import StandardScaler

std=StandardScaler().fit(x)
x_std=std.transform(x)


# In[60]:


x_std


# In[61]:


x_train_std,x_test_std,y_train,y_test=train_test_split(x_std,y,random_state=10,test_size=0.3,shuffle=True)


# In[62]:


x_train_std.shape


# In[63]:


k_range=range(1,45)
scores={}
h_score = 0     
best_k=0        
scores_list=[]  

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_std,y_train)
    prediction_knn=knn.predict(x_test)
    scores[k]=accuracy_score(y_test,prediction_knn)
    if scores[k]>h_score:
        h_score = scores[k]
        best_k = k

    scores_list.append(accuracy_score(y_test,prediction_knn))
print('The best value of k is {} with score : {}'.format(best_k,h_score))


# In[64]:


scores


# In[65]:


plt.plot(k_range,scores_list)


# In[ ]:





# In[66]:


knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train_std,y_train)


# In[ ]:





# In[67]:


prediction_knn=knn.predict(x_test_std)
accuracy_knn=accuracy_score(y_test,prediction_knn)*100
print('accuracy_score score     : ',accuracy_score(y_test,prediction_knn)*100,'%')
print('mean_squared_error score : ',mean_squared_error(y_test,prediction_knn)*100,'%')


# In[68]:


Category=['No,You donot have Heart disease','Sorry ,You are having heart disease']


# In[69]:


custom_data_knn=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
custom_data_knn_std=std.transform(custom_data_knn)
custom_data_prediction_knn=knn.predict(custom_data_knn_std)


# In[70]:


custom_data_prediction_knn


# In[71]:


print(Category[int(custom_data_prediction_knn)])


# In[72]:


from sklearn.svm import SVC

svc= SVC(C=2.0,kernel='rbf',gamma='auto').fit(x_train_std,y_train)
Y_predict = svc.predict(x_test_std)
accuracy_svc=accuracy_score(y_test,Y_predict)*100


# In[73]:


print('Accuracy score : {}%'.format(accuracy_score(y_test,Y_predict)*100))


# In[74]:


custom_data_svc=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
custom_data_svc_std=std.transform(custom_data_svc)
custom_data_prediction_svc=svc.predict(custom_data_svc_std)


# In[75]:


custom_data_prediction_svc


# In[76]:


print(Category[int(custom_data_prediction_svc)])


# In[77]:


from lightgbm import LGBMClassifier

lg=LGBMClassifier(boosting_type='gbdt',n_estimators=24,learning_rate=0.25,objective='binary',metric='accuracy',is_unbalance=True,
                 colsample_bytree=0.7,reg_lambda=3,reg_alpha=3,random_state=500,n_jobs=-1,num_leaves=20)
lg.fit(x_train,y_train)
yprid = lg.predict(x_test)
accuracy_lgbm=accuracy_score(y_test,yprid)*100


# In[78]:


print('With score : ',accuracy_score(y_test,yprid)*100)


# In[79]:


custom_data_lgbm=np.array([[61 ,1, 3,141,223,1,1,140,0,2.3,0,0,1]])
custom_data_prediction_lgbm=lg.predict(custom_data_lgbm)


# In[80]:


custom_data_prediction_lgbm


# In[81]:


print(Category[int(custom_data_prediction_lgbm)])


# In[82]:


from sklearn.ensemble import RandomForestClassifier
ranm = RandomForestClassifier(n_estimators=100, bootstrap = True)
ranm.fit(x_train,y_train)


# In[83]:


ranpr = ranm.predict(x_test)
accuracy_random=accuracy_score(y_test,ranpr)*100


# In[84]:


print('With score : ',accuracy_random)


# In[85]:


custom_data_random=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
custom_data_prediction_random=ranm.predict(custom_data_random)


# In[86]:


custom_data_prediction_random


# In[87]:


print(Category[int(custom_data_prediction_random)])


# In[88]:


algorithms=['Decision Tree','KNN','SVC','LGBM','RANDOM FOREST']
scores=[accuracy_dt,accuracy_knn,accuracy_svc,accuracy_lgbm,accuracy_random]


# In[ ]:





# In[ ]:





# In[ ]:





# In[89]:


sns.barplot(algorithms,scores)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




