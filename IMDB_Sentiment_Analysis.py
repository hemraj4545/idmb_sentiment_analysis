#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>Logistic Regression : Sentiment Analysis Case Study</h1>

# - IMDB movie reviews dataset
# - http://ai.stanford.edu/~amaas/data/sentiment
# - Contains 25000 positive and 25000 negative reviews
# <img src="https://i.imgur.com/lQNnqgi.png" align="center">
# - Contains at most reviews per movie
# - At least 7 stars out of 10 $\rightarrow$ positive (label = 1)
# - At most 4 stars out of 10 $\rightarrow$ negative (label = 0)
# - 50/50 train/test split
# - Evaluation accuracy
# <b>Features: bag of 1-grams with TF-IDF values</b>:
# - Extremely sparse feature matrix - close to 97% are zeros
# 
#  <b>Model: Logistic regression</b>
# - $p(y = 1|x) = \sigma(w^{T}x)$
# - Linear classification model
# - Can handle sparse data
# - Fast to train
# - Weights can be interpreted
# <img src="https://i.imgur.com/VieM41f.png" align="center" width=500 height=500>

# ## Introduction and Importing of Data

# In[9]:


import pandas as pd

data = pd.read_csv('movie_data.csv')
data.head(10)


# In[10]:


data.shape


# ## Transforming Documents into Feature Vectors
# **Bag of words/Bag of N-words model**
# 
# Below, we will call the fit_transform method on CountVectorizer. This will construct the vocabulary of the bag-of-words model and transform the following three sentences into sparse feature vectors:
# 1. The sun is shining
# 2. The weather is sweet
# 3. The sun is shining, the weather is sweet, and one and one is two
# 

# In[11]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array(['The sun is shining.',
               'The weather is sweet.',
               'The sun is shining, the weather is sweet, and one and one is two.'])

bag = count.fit_transform(docs)


# In[12]:


print(count.vocabulary_)


# In[13]:


print(bag.toarray())


# Raw term frequencies: *tf (t,d)*—the number of times a term t occurs in a document *d*

# ## Term Frequency-Inverse Document Frequency (TF-IDF)

# $$\text{tf-idf}(t,d)=\text{tf (t,d)}\times \text{idf}(t,d)$$
# 
# $$\text{idf}(t,d) = \text{log}\frac{n_d}{1+\text{df}(d, t)},$$
# 
# where $n_d$ is the total number of documents, and df(d, t) is the number of documents d that contain the term t.

# In[19]:


from sklearn.feature_extraction.text import TfidfTransformer

np.set_printoptions(precision=2)
tfid = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
print(tfid.fit_transform(count.fit_transform(docs)).toarray())


# The equations for the idf and tf-idf that are implemented in scikit-learn are:
# 
# $$\text{idf} (t,d) = log\frac{1 + n_d}{1 + \text{df}(d, t)}$$
# The tf-idf equation that is implemented in scikit-learn is as follows:
# 
# $$\text{tf-idf}(t,d) = \text{tf}(t,d) \times (\text{idf}(t,d)+1)$$
# 
# $$v_{\text{norm}} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v_{1}^{2} + v_{2}^{2} + \dots + v_{n}^{2}}} = \frac{v}{\big (\sum_{i=1}^{n} v_{i}^{2}\big)^\frac{1}{2}}$$
# 
# ### Example:
# $$\text{idf}("is", d3) = log \frac{1+3}{1+3} = 0$$
# Now in order to calculate the tf-idf, we simply need to add 1 to the inverse document frequency and multiply it by the term frequency:
# 
# $$\text{tf-idf}("is",d3)= 3 \times (0+1) = 3$$

# In[57]:


tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1) / (3+1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term is = %.2f' % tfidf_is)


# $$\text{tfi-df}_{norm} = \frac{[3.39, 3.0, 3.39, 1.29, 1.29, 1.29, 2.0 , 1.69, 1.29]}{\sqrt{[3.39^2, 3.0^2, 3.39^2, 1.29^2, 1.29^2, 1.29^2, 2.0^2 , 1.69^2, 1.29^2]}}$$$$=[0.5, 0.45, 0.5, 0.19, 0.19, 0.19, 0.3, 0.25, 0.19]$$$$\Rightarrow \text{tfi-df}_{norm}("is", d3) = 0.45$$

# In[58]:


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
raw_tfidf


# In[59]:


l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf


# ## Data Preparation

# In[22]:


data.loc[0,'review'][-50:]


# In[23]:


import re

def func(text):
    text = re.sub('<[^>]*>','',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower()) +        ' '.join(emoticons).replace('-','')
    return text


# In[24]:


func(data.loc[0,'review'][-50:])


# In[25]:


data['review'] = data['review'].apply(func)


# ## Tokenization of Documents

# In[28]:


from nltk.stem.porter import PorterStemmer

Porter = PorterStemmer()


# In[29]:


def tokenizer(text):
    return text.split()


# In[30]:


def tokenizer_stemmer(text):
    return [Porter.stem(word) for word in text.split()]


# In[33]:


tokenizer('I am good and bad at maths, so I enjoying doing work')


# In[34]:


tokenizer_stemmer('I am good and bad at maths, so I enjoying doing work')


# In[35]:


import nltk
nltk.download('stopwords')


# In[36]:


from nltk.corpus import stopwords

stop = stopwords.words('english')
[i for i in tokenizer_stemmer('I am good and bad at maths, so I enjoying doing work')[-10:] if i not in stop]


# ## Transform Text Data into TF-IDF Vectors

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=None,
                       tokenizer = tokenizer_stemmer,
                       use_idf=True,
                       smooth_idf=True,
                       norm = 'l2')


# In[39]:


y = data.sentiment.values
x = tfidf.fit_transform(data['review'])


# ## Document Classification using Logistic Regression

# In[41]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)


# In[43]:


import pickle
from sklearn.linear_model import LogisticRegressionCV

clf = LogisticRegressionCV(cv=5,
                          scoring='accuracy',
                          random_state=0,
                          n_jobs=-1,
                          verbose=3,
                          max_iter=300).fit(x_train,y_train)

model = open('model.sav','wb')
pickle.dump(clf,model)
model.close()


# ## Model Evaluation

# In[46]:


model = 'model.sav'
clf = pickle.load(open(model,'rb'))


# In[51]:


prediction = clf.score(x_test,y_test)


# In[53]:


print("Prediction = {0}%".format(prediction*100))

