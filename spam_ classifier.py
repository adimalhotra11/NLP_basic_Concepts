# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:34:52 2020

@author: Aditya
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
import re
df = pd.read_csv('SMSSpamCollection.txt',sep='\t',names=['label','message'],index_col=False)
wordnet = WordNetLemmatizer()
ps = PorterStemmer()
corpus = []
for i in range(len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000)

X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])

y = y.iloc[:,1].values


from sklearn.model_selection import train_test_split

X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train,y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test,y_pred)