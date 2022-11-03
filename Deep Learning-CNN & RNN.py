#!/usr/bin/env python
# coding: utf-8

# ### Name : Berchmans Kevin S
# 

# ## `PDL CNN and RNN`

# In[1]:


# Necessary Packages

import numpy as np 
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Dropout, Flatten
from keras.layers import Embedding, SimpleRNN,LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')


# In[2]:


from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# # `Model - RNN`

# In[3]:


df = pd.read_csv("Train.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


x = df['text']
y = df['label']


# In[9]:


y.value_counts()


# In[10]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ly= le.fit(y)


# In[11]:


Y = le.fit_transform(y)


# In[12]:


Y


# In[13]:


# Splitting

X_train,X_test, y_train, y_test = train_test_split(x,Y,test_size=0.2,random_state=42)
X_train


# In[14]:


# Preprocessing

train_token = Tokenizer(num_words=100,oov_token='<oov>')
train_token.fit_on_texts(X_train)
word_index = train_token.word_index
train_sequence = train_token.texts_to_sequences(X_train)
dict(list(word_index.items())[0:20])


# In[15]:


vocab = len(train_token.word_index) + 1
vocab


# In[16]:


train_sequence[25]


# In[17]:


train_padded = pad_sequences(train_sequence,maxlen=100,padding='post')


# In[18]:


train_padded.shape


# In[19]:


train_padded[5]


# In[20]:


val_token = Tokenizer(num_words=500,oov_token='<oov>')
val_token.fit_on_texts(X_test)
val_index = val_token.word_index
val_sequence = val_token.texts_to_sequences(X_test)


# In[21]:


val_padded = pad_sequences(val_sequence,maxlen=100,padding='post')


# In[22]:


modelr = Sequential()
# Embedding layer
modelr.add(Embedding(300,70,input_length=100))
modelr.add(SimpleRNN(70,activation='relu'))
modelr.add(Dense('1',activation='sigmoid'))


# In[23]:


modelr.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[24]:


modelr.summary()


# In[25]:


history1 = modelr.fit(train_padded,y_train,epochs=5,verbose=2,batch_size=15)


# In[26]:


plt.plot(history1.history['accuracy'])
plt.title('model loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


# In[27]:


modelr1 = Sequential()
# Embedding layer
modelr1.add(Embedding(5000,64,input_length=100))
modelr1.add(SimpleRNN(32,activation='tanh'))
modelr1.add(Embedding(5000,32,input_length=100))
modelr1.add(SimpleRNN(32,activation='tanh' ))
modelr1.add(Dense('1',activation='sigmoid'))


# In[28]:


modelr1.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[29]:


modelr1.summary()


# In[30]:


history2 = modelr1.fit(train_padded,y_train,epochs=5,verbose=2,batch_size=15)


# In[31]:


plt.plot(history2.history['accuracy'])
plt.title('model loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend(['train', 'validation'])
plt.show()


# # `Model - CNN`

# In[32]:


from keras.preprocessing.image import ImageDataGenerator


# In[33]:


# Train_data

train_datagen=ImageDataGenerator(rescale=0.2,horizontal_flip=True,zoom_range=0.2,shear_range=0.2)

train_data = train_datagen.flow_from_directory(directory="Cars Dataset/train/")


# In[34]:


train_data.class_indices


# In[35]:


# Model building

model=Sequential()
model.add(Conv2D(filters=32,input_shape=(256,256,3),kernel_size=(3,3),activation='relu'))

model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(rate=0.50))
model.add(Dense(1,activation='sigmoid'))


# In[36]:


# Model Compile

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[37]:


model.summary()


# In[39]:


model.fit(train_data,epochs=1)


# In[ ]:




