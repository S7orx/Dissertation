#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install langdetect


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from collections import Counter
from bs4 import BeautifulSoup
import seaborn as sns
from langdetect import DetectorFactory, detect


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


nltk.download('stopwords')
nltk.download('punkt')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wn = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[4]:


df_twcs = pd.read_csv("/Users/saharawaji/Documents/GitHub/Research/twcs 2.csv", on_bad_lines='skip')
df_twcs


# In[5]:


column_names = df_twcs.columns
print(column_names)


# ### Exploratory Data Analysis (EDA)

# In[6]:


# Show information about the dataset
print(df_twcs.info())

#Summary statistics of numerical columns
print(df_twcs.describe())

# Check the first few rows of the dataset
print(df_twcs.head())


# ### Handling Missing Data:
# 
# 

# In[7]:


# Drop rows with missing values
df_twcs.dropna(inplace=True)


# In[8]:


# Impute missing values in the 'tweet_id' column with the mean
df_twcs['tweet_id'].fillna(df_twcs['tweet_id'].mean(), inplace=True)


# In[9]:


# Count the number of missing values in each column
missing_values_count = df_twcs.isna().sum()


# print count of missing values for each column
print(missing_values_count)


# In[10]:


df_twcs


# In[11]:


X = 2811774 - 976810

X


# ### Text Preprocessing:
# 
# 

# In[12]:


# Tokenization
df_twcs['text'] = df_twcs['text'].apply(lambda x: word_tokenize(x.lower()))

# Stop word removal
stop_words = set(stopwords.words('english'))
df_twcs['text'] = df_twcs['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Stemming (or Lemmatization)
stemmer = PorterStemmer()

df_twcs['text'] = df_twcs['text'].apply(lambda tokens: [word.lower() for word in tokens])


# In[13]:


df_twcs['text']


# # Feature Engineering:
# 
# 1. Text Features:
# 2. Temporal Features:
# 3. Categorical Features:
# 4. Text-Based Features (NLP):
# 5. Interaction Features:
# 6. Domain-Specific Features:
# 7. Feature Scaling:
# 8. Feature Selection:
# 

# In[14]:


# 1. Text Features:
# A. Text Length: Create a feature that represents the length of the text in characters or words.
df_twcs['text_length'] = df_twcs['text'].apply(len)
print(df_twcs['text_length'])

# B. Word Count: Count the number of words in each text.
df_twcs['word_count'] = df_twcs['text'].apply(lambda tokens: len(tokens))

print(df_twcs['word_count'])


# In[18]:


#2. Features of Temporal Order:

# A. Extracting Date and Time: Parse 'created_at' if it is in a string format to extract relevant temporal information such as month, day, hour, and so on.

#df_twcs['created_at'] = pd.to_datetime(df_twcs['created_at'])
df_twcs['created_at'] = pd.to_datetime(df_twcs['created_at'], utc=True)


df_twcs['year'] = df_twcs['created_at'].dt.year
df_twcs['month'] = df_twcs['created_at'].dt.month

# B. Time Since Tweet: Calculate the time elapsed since a tweet was posted relative to a reference date.
#reference_date = pd.Timestamp('2024-01-31')
reference_date = pd.Timestamp('2024-01-31', tz='UTC')
f_twcs['time_since_posted'] = (reference_date - df_twcs['created_at']).dt.total_seconds()




# In[ ]:


# 3. Categorical Features:
# Encoding Categorical Variables: Convert categorical variables like 'author_id' into numerical values using techniques like one-hot encoding or label encoding.

df_twcs = pd.get_dummies(df_twcs, columns=['author_id'], prefix='author')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




