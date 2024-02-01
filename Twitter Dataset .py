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


# Display basic information about the dataset
print(df_twcs.info())

# Get summary statistics of numerical columns
print(df_twcs.describe())

# Check the first few rows of the dataset
print(df_twcs.head())


# ### Handling Missing Data:
# 
# 

# In[7]:


# Drop rows with missing values (use with caution)
df_twcs.dropna(inplace=True)


# In[8]:


# Impute missing values in the 'tweet_id' column with the mean
df_twcs['tweet_id'].fillna(df_twcs['tweet_id'].mean(), inplace=True)


# In[9]:


# Count the number of missing values in each column
missing_values_count = df_twcs.isna().sum()

# Alternatively, you can use isnull() instead of isna()
# missing_values_count = df_twcs.isnull().sum()

# Display the count of missing values for each column
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
#df_twcs['word_count'] = df_twcs['text'].apply(lambda x: len(x.split()))
# Count the number of words in each text (assuming 'text' is a list of tokens)
df_twcs['word_count'] = df_twcs['text'].apply(lambda tokens: len(tokens))

print(df_twcs['word_count'])

# C. Presence of Keywords: Create binary features indicating the presence of specific keywords in the text.
keywords = ['important', 'urgent', 'help']
for keyword in keywords:
    df_twcs[keyword + '_present'] = df_twcs['text'].apply(lambda x: 1 if keyword in x else 0)


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


#df_twcs['time_since_posted'] = (reference_date - df_twcs['created_at']).dt.total_seconds()
#df_twcs['time_since_posted'] = (reference_date - df_twcs['created_at']).dt.total_seconds()
df_twcs['time_since_posted'] = (reference_date - df_twcs['created_at']).dt.total_seconds()




# In[ ]:


# 3. Categorical Features:
# Encoding Categorical Variables: Convert categorical variables like 'author_id' into numerical values using techniques like one-hot encoding or label encoding.


df_twcs = pd.get_dummies(df_twcs, columns=['author_id'], prefix='author')


# In[ ]:


# 4. Text-Based Features (NLP):
# A. TF-IDF: Compute TF-IDF scores for words in the text to represent their importance.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_twcs['text'])

# B. Word Embeddings: Use pre-trained word embeddings (e.g., Word2Vec, GloVe) to convert text into dense vectors.

# Load pre-trained Word2Vec model
word2vec_model = Word2Vec.load('word2vec.model')

def text_to_embedding(text):
    words = text.split()
    vectors = [word2vec_model[word] for word in words if word in word2vec_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

df_twcs['text_embedding'] = df_twcs['text'].apply(text_to_embedding)


# In[ ]:


# 5. Interaction Features:
# Create interaction features by combining existing features.

df_twcs['text_length_word_count_ratio'] = df_twcs['text_length'] / df_twcs['word_count']


# In[ ]:


# 6. Domain-Specific Features:

# If you have domain-specific knowledge, create features that are relevant to your specific problem.


# In[ ]:


#7. Feature Scaling:

# Normalize or scale numerical features if needed using techniques like Min-Max scaling or Standardization.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_twcs[['text_length', 'word_count']] = scaler.fit_transform(df_twcs[['text_length', 'word_count']])


# In[ ]:


# 8. Feature Selection:

# Perform feature selection techniques (e.g., Recursive Feature Elimination, feature importance from tree-based models) to choose the most relevant features for your model.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


inbound_chat = df_twcs[df_twcs.inbound]

inbound_chat


# In[14]:


df_in_outs = pd.merge(inbound_chat, df_twcs, left_on='tweet_id', 
                                  right_on='in_response_to_tweet_id')

df_in_outs


# In[15]:


class Preprocess_Dataset:
    def __init__(self, dataframe):
        self.df = dataframe
        self.last_id = 0
        self.conv = []
        self.company_name = ''
        self.df_convs = pd.DataFrame(columns=['author_id', 'company_name', 'dialog'])
        
    def add_to_df(self, last_id, author_id, company_name, text_x, text_y):
        if (last_id == author_id):
            self.conv.append('participant1|'+ " ".join(filter(lambda x:x[0]!='@', text_x.split())))
            self.conv.append('participant2|'+ " ".join(filter(lambda x:x[0]!='@', text_y.split())))
        elif self.last_id != 0:
            if len(self.conv) > 0:
                id = len(self.df_convs)
                self.df_convs.loc[id, 'author_id'] = self.last_id
                self.df_convs.loc[id, 'company_name'] = self.company_name
                self.df_convs.loc[id, 'dialog'] = self.conv
                self.conv = []
            self.last_id = author_id
            
        else:
            self.conv.append('participant1|'+ text_x)
            self.conv.append('participant2|'+ text_y)
            self.last_id = author_id
            self.company_name = company_name
    def create_df(self):
        [self.add_to_df(self.last_id, row[0], row[1], row[2], row[3]) for row in self.df[['author_id_x', 'author_id_y','text_x', 'text_y']].values]
        return self.df_convs


# In[16]:


def clean_text(text):
    clean_text = []
    text = re.sub("''", "", text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split())) # Remove the words starts with '@'
    text = re.sub("(\\d|\\W)+"," ", text)
    clean_text = [wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) 
                  if (word not in stop_words and word not in list(string.punctuation))]
    return clean_text
    #return " ".join([word for word in clean_text])


# In[17]:


def split_participants(conversation):
    part1_dialog = []
    part2_dialog = []
    conv_token = []
    for conv in conversation:
        dialog = conv.split('|')
        if dialog[0] == 'participant1':
            part1_dialog.append(dialog[1])
        else:
            part2_dialog.append(dialog[1])
            
    if (len(part1_dialog) > 0):
        part1_str = " ".join([word for word in part1_dialog])
        conv_token.append(clean_text(part1_str))
    if (len(part2_dialog) > 0):
        part2_str = " ".join([word for word in part2_dialog])
        conv_token.append(clean_text(part2_str))
    return conv_token


# In[ ]:




