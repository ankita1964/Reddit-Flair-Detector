import pickle
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

india_data = pd.read_csv("reddit-india-data.csv")

flairs = india_data.flair.unique()

replace_by_space = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
stop_words = stopwords.words('english')

def clean_data(text):
    #converting to lowercase
    text = text.lower()
    #re.sub(new_value, text_to_processed) 
    text = replace_by_space.sub(' ', text)
    text = bad_symbols.sub('', text)
    #removing the stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words) 
    return text

def clean_url(u):
    if u.startswith("http://"):
        u = u[7:]
    if u.startswith("https://"):
        u = u[8:]
    if u.startswith("www."):
        u = u[4:]
    if u.endswith("/"):
        u = u[:-1]
    return u

def reddit_url(u):
    u = u.replace('redditcom', '')
    u = u.replace('r', '')
    u = u.replace('india', '')
    u = u.replace('comments','')
    for word in u:
        u = ' '.join(u.split('_'))
    return u

india_data['title'] = india_data['title'].apply(clean_data)
india_data['comments'] = india_data['comments'].astype('str').apply(clean_data)
india_data['body'] = india_data['body'].astype('str').apply(clean_data)

india_data['url'] = india_data['url'].apply(clean_url)
india_data['url'] = india_data['url'].apply(clean_data)

india_data['url'] = india_data['url'].apply(reddit_url)
india_data['url'] = india_data['url'].apply(clean_data)

india_data_impute = india_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
india_data_impute['title_comments_body_url'] = india_data_impute['title'] + '  ' + india_data_impute['comments'] + '  ' + india_data_impute['body'] + '  ' + india_data_impute['url']

X = india_data_impute.title_comments_body_url
y = india_data_impute.flair

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1) 

logreg = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', LogisticRegression(n_jobs=1, C=1e5))])
logreg.fit(X_train, y_train)

pickle.dump(logreg, open('lr_model.pkl', 'wb'))

y_pred = logreg.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=flairs))