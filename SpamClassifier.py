# -*- coding: utf-8 -*-
"""
Created on Thu May 14 12:19:23 2020

@author: smith
"""
import pandas as pd 

messages = pd.read_csv("SMSSpamCollection", sep='\t', names = ["label", "message"])

#Data cleaning and preprocessing
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(messages)):
    reviews = re.sub("[^a-zA-Z]", ' ', messages['message'][i])
    reviews = reviews.lower()
    reviews = reviews.split()
    reviews = [wordnet.lemmatize(word) for word in reviews if word not in set(stopwords.words('english'))]
    reviews = " ".join(reviews)
    corpus.append(reviews)
    
#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()     


#Creating a TF-IDF Model
#from sklearn.feature_extraction.text import TfidfVectorizer
#cv = TfidfVectorizer()
#X = cv.fit_transform(corpus).toarray()     

    
Y = pd.get_dummies(messages['label'])
Y = Y.iloc[:,1].values    
    
#Creating Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)    
  
#Training Model using Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, Y_train)  
    
#Prediting Output
Y_pred = model.predict(X_test)

#Confusion metrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test, Y_pred)
    
    
    