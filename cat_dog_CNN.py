#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf


# In[41]:


DATADIR = "D:\Projects_Code\kagglecatsanddogs_3367a\PetImages"
from tensorflow.keras.callbacks import TensorBoard
import time
NAME = "Cats-vs-Dogs-bnn-64x2-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='log/{}'.format(NAME))


# In[39]:


CATEGORIES = ["Dog","Cat"]


# In[10]:


print(img_array.shape)


# In[12]:





# In[15]:


training_data = []

def create_training_data():
    
    for  category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(50,50))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data()


# In[17]:


print(len(training_data))


# In[18]:


import random
random.shuffle(training_data)
    


# In[19]:


for sample in training_data[:10]:
    print(sample[1])


# In[20]:


X = []
y = []


# In[21]:


for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1,50,50,1)    


# In[27]:


import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()


# In[28]:


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)


# In[29]:


X[1]


# In[32]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation,Flatten, Conv2D, MaxPooling2D


# In[42]:


X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

model = Sequential()


model.add(Conv2D((64),(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D((64),(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer = "adam",metrics=["accuracy"])

model.fit(X,y,batch_size=32,epochs=10,validation_split=0.1,callbacks=[tensorboard])



# In[35]:





# In[ ]:




