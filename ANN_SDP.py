
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('kc2.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 21].values

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[2]:


import keras
from keras.models import Sequential
from keras.layers import Dense



# In[3]:


classifier = Sequential()


# In[4]:


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 21))


# In[5]:


classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[6]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[7]:


classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# In[8]:



y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.3)


# In[19]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[11]:


#cm
y_pred


# In[12]:


y_test


# In[15]:


y_pred = (y_pred > 0.5)


# In[16]:


sum(y_test)


# In[45]:


for i in range(len(y_pred)):
    print(i,y_pred[i],y_test[i])

