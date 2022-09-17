#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())


# In[3]:


pwd()


# In[63]:


df = pd.read_excel('datasets/Tweet_NFT.xlsx')   #Read the file


# In[64]:


df.head()  #check first 5 rows


# In[65]:


#we require only text and tweet intent column so we drop other columns
df.drop(['id','tweet_created_at'],axis=1,inplace=True)


# In[66]:


df.head()


# In[67]:


df.isnull().sum()  #check missing values


# In[68]:


df.dropna(inplace=True)   #drop them


# In[69]:


df['tweet_intent'].value_counts().plot(kind='bar')   #this shows the count of each tweet intent


# In[76]:


#cleaning tweets
def clean_tweet(tweet):
    tweet = ''.join([c for c in tweet if ord(c) < 128])
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = re.sub('!', "",tweet)
    tweet = re.sub('-', "",tweet)
    tweet = re.sub(':', "",tweet)
    tweet = re.sub('" "', "",tweet)
    tweet = tweet.replace("'", "")
    tweet = tweet.lower()
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)          if w.lower() in words or not w.isalpha())
    return tweet


# In[77]:


df['tweet_text'] = df['tweet_text'].map(lambda x: clean_tweet(x))
df


# ### Vectorize tweet_text, by turning each text into either a sequence of integers or into a vector.Limit the data set to the top 50000 words.Set the max number of words in each tweet at 250.

# In[83]:


# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each tweet.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['tweet_text'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# ### Truncate and pad the input sequences so that they are all in the same length for modeling.

# In[84]:


X = tokenizer.texts_to_sequences(df['tweet_text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)


# ### Converting categorical labels to numbers.

# In[85]:


Y = pd.get_dummies(df['tweet_intent']).values
print('Shape of label tensor:', Y.shape)


# ### Train test split.

# In[86]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# ### The first layer is the embedded layer that uses 100 length vectors to represent each word.SpatialDropout1D performs variational dropout in NLP models.The next layer is the LSTM layer with 100 memory units.The output layer must create 13 output values, one for each class.Activation function is softmax for multi-class classification.Because it is a multi-class classification problem, categorical_crossentropy is used as the loss function.

# In[89]:


model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[90]:


epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


# In[91]:


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[92]:


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();


# In[97]:


plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();


# ### The plots suggest that the model has a little over fitting problem, more data may help, but more epochs will not help using the current data.

# In[ ]:




