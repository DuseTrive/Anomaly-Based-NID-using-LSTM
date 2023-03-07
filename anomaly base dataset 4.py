#!/usr/bin/env python
# coding: utf-8

# # import pands modules

# In[ ]:





# In[1]:


import pandas as pd
from glob import glob
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt


# # Import dataset 

# # !!!!! the datset contin more than 3GB dont run it on computer less than 32GB ram !!!!!!!!!!!!

# In[2]:


df = pd.read_csv("data/data_4.csv") #CSE-CIC-IDS 2018 (02-20-2018)
df


# # replace space with "_"

# In[4]:


cols = df.columns
cols = cols.map(lambda x: x.replace(' ', '_') )
df.columns = cols


# # filter HTTP and HTTPS traffic

# In[5]:


query = df.query('Dst_Port == 80 or Dst_Port == 443')
df=query
df


# # Checking for null and INFINITE

# In[6]:


#check for null
df.isnull().any()


# In[7]:


# counting infinity in a particular column name
inf=df.isin([np.inf, -np.inf])
inf


# In[8]:


#replace infinit number
df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
##df = df[np.isfinite(df).all(1)]
#drop null
df.dropna(how = 'all')


# In[ ]:





# In[ ]:





# In[ ]:





# # Checking Data type

# In[9]:


print('Data type of each column of Dataframe :')
df.info(verbose=True)


# # Drop all column contain object datatype

# In[10]:


df = df.drop(columns=['Timestamp', 'Flow_ID', 'Src_IP', 'Dst_IP'])
df


# In[11]:


daummy = pd.get_dummies(df['Label'])
daummy.head()


# In[12]:


#change values as
#benign = 0
#DDoS attacks-LOIC-HTTP = 1
daummy.head()
df.Label[df.Label=='Benign'] = 0
df.Label[df.Label =='DDoS attacks-LOIC-HTTP'] = 1


# In[13]:


daummy = pd.get_dummies(df['Label'])
daummy.head()


# # date devision

# In[14]:


bening_df = df[df['Label']==0][0:1332938]
malignant_df = df[df['Label']==1][0:2665876]


# In[15]:


axes = bening_df.plot(kind='scatter', x='Flow_Duration', y = 'Tot_Fwd_Pkts', color='blue', label='Benign')
malignant_df.plot(kind='scatter', x='Flow_Duration', y = 'Tot_Fwd_Pkts', color='red', label='maligmant', ax=axes)


# # Create Raw and label vaibles

# In[16]:


#limiting number of rows in order to speed up the process
#first shuffel raws
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
number_of_data=15000# can chanhe in between 500- 2665877 
train_df = df [0:number_of_data]
train_df = train_df.astype("float64")
train_df 


# In[17]:


##Removing unwanted columns 
df.columns 

feature_df = train_df[['Src_Port', 'Dst_Port', 'Protocol', 'Flow_Duration', 'Tot_Fwd_Pkts',
       'Tot_Bwd_Pkts', 'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts', 'Fwd_Pkt_Len_Max',
       'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std',
       'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean',
       'Bwd_Pkt_Len_Std', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max',
       'Flow_IAT_Min', 'Fwd_IAT_Tot', 'Fwd_IAT_Mean', 'Fwd_IAT_Std',
       'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_IAT_Tot', 'Bwd_IAT_Mean',
       'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags',
       'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Len',
       'Bwd_Header_Len', 'Fwd_Pkts/s', 'Bwd_Pkts/s', 'Pkt_Len_Min',
       'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var',
       'FIN_Flag_Cnt', 'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt',
       'ACK_Flag_Cnt', 'URG_Flag_Cnt', 'CWE_Flag_Count', 'ECE_Flag_Cnt',
       'Down/Up_Ratio', 'Pkt_Size_Avg', 'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg',
       'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg',
       'Bwd_Byts/b_Avg', 'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg',
       'Subflow_Fwd_Pkts', 'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts',
       'Subflow_Bwd_Byts', 'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts',
       'Fwd_Act_Data_Pkts', 'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std',
       'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max',
       'Idle_Min']]


#df 2665877 rows and 80 columns,
#pick 79 columns
# pick 15000 rows
#independent var
raw = np.asarray(feature_df)
#dependent varinable
label= np.asarray(train_df['Label'])


# In[18]:


#np.all(np.isfinite(x))
#np.all(np.isfinite(y))


# In[19]:


#y = y[np.isfinite(df).all(1)]


# In[20]:


#df.replace([np.inf, -np.inf], np.nan)

#df.dropna(inplace=True)


# In[21]:


#y = y.reset_index()
raw


# In[ ]:





# # Data Devision

# In[22]:


## import deviding model
from sklearn.model_selection import train_test_split
raw_train,raw_test,label_train,label_test=train_test_split(raw,label, test_size=0.2, random_state=4)


# In[ ]:





# # SVM Model

# In[23]:


##test usign knowladge and feeling of testing data 
from sklearn import svm 
model=svm.SVC(kernel="rbf", C=3, coef0=0.001, degree =10, gamma='scale')
model.fit(raw_train,label_train)
acc = model.score(raw_test, label_test)
print("Anomaly based IDs Accuracy with SVM: {:.2f}%".format(acc * 100))


# In[24]:


##use grid function to find best matching parameters
#from sklearn.model_selection import GridSearchCV 
#param= {'kernel':('linear','poly','rbf','sigmoid'),
#        'C':[2,3,1],
#        'degree':[3,100],
#        'coef0':[0.001,1000,0.5],
#        'gamma': ('auto', 'scale')
#       }
#SVModel=svm.SVC()
#GridS = GridSearchCV(SVModel, param, cv=5)
#GridS.fit(raw_test, label_test)
#GridS.best_params_


# In[25]:


##implemet and optimize wtih grid serch 
#model=svm.SVC(kernel="linear", C=2, coef0=0.001, degree =3, gamma='auto')
#model.fit(raw_train,label_train)
#acc = model.score(raw_test, label_test)
#print("Anomaly based IDs Accuracy with SVM: {:.2f}%".format(acc * 100))


# In[26]:


#plot_confusion_matrix(estimator, X, y_true) 
from sklearn.metrics import plot_confusion_matrix   
plot_confusion_matrix(model, raw_test, label_test) 


# In[27]:


## Generate classification report 
from sklearn.metrics import classification_report
svm_prediction = model.predict(raw_test)
print( '\nClasification report:\n', classification_report(label_test, svm_prediction))


# # Long Short Term Memory (LSTM)

# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


# In[29]:


# reshape input to be [samples, time steps, features]
raw_train = np.reshape(raw_train, (raw_train.shape[0], 1, raw_train.shape[1]))
raw_test = np.reshape(raw_test, (raw_test.shape[0], 1,raw_test.shape[1]))


# In[30]:


raw_train.shape


# In[31]:


# not working execute with error 
####look_back = 1
###model = Sequential()
###model.add(LSTM(4, input_shape=(1, look_back)))
###model.add(Dense(1))
###model.compile(loss='mean_squared_error', optimizer='adam')
##model.fit(raw_train, label_train, epochs=100, batch_size=1, verbose=2)
####model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
#####model.fit(df, target, nb_epoch=10000, batch_size=1, verbose=2,validation_data=(x_test, y_test))


# In[ ]:





# In[32]:


regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])
history= regressor.fit(raw_train, label_train, epochs=15, batch_size=1, verbose=2,validation_data=(raw_test, label_test))
history 


# In[33]:


regressor.summary()


# In[34]:


#results = regressor.predict(raw_test)
#results


# In[35]:


#results=np.arange(0,len(results),1)
#results.shape


# In[36]:


#label_test.shape


# In[ ]:





# In[39]:


#plt.scatter(range(20),results[:,0],c='r')
#plt.scatter(range(20),label_test[:,0],c='r')
#plt.show()


# In[38]:


#loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['loss'], 'o')
plt.plot(history.history['val_loss'], 'o')
plt.plot(history.history['val_loss'])
plt.title('comparing predict and real value loss presentage')
plt.xlabel('Loss persentage')
plt.ylabel('Epoch')
plt.legend(['Predict_loss','Predict_loss poins','Real_loss points', 'Real_loss'])
plt.show


# In[40]:


#accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['accuracy'],'o')
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_accuracy'],'o')
plt.title('comparing predict and real accuracy presentage')
plt.xlabel('Accuracy persentage')
plt.ylabel('Epoch')
plt.legend(['Predict_Accuracy','Predict_Accuracy points', 'Real_accuracy','Real_accuracy points'])
plt.show


# In[41]:


#accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['accuracy'],'o')
plt.title('predict accuracy')
plt.show


# In[42]:


#val_accuracy Plot
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['val_accuracy'],'o')
plt.title('real accuracy')
plt.show


# In[ ]:




