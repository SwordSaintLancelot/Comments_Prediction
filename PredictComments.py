# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 12:35:10 2020

@author: Ghost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
import nltk

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB


train_data = pd.read_csv(r'E:\GitHub\data\Comments_Prediction\train.csv')
test_data = pd.read_csv(r'E:\GitHub\data\Comments_Prediction\test.csv')
test_labels = pd.read_csv(r'E:\GitHub\data\Comments_Prediction\test_labels.csv')

train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\n", " ", x))
train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\"", "", x))
# train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\"", "", x))
# train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\"", "", x))

test_data['comment_text'] = test_data['comment_text'].map(lambda x: re.sub("\n", " ", x))
test_data['comment_text'] = test_data['comment_text'].map(lambda x: re.sub("\"", "", x))
# test_data['comment_text'] = test_data['comment_text'].map(lambda x: re.sub("\n", " ", x))
# test_data['comment_text'] = test_data['comment_text'].map(lambda x: re.sub("\n", " ", x))

x = train_data['comment_text']
y = train_data.iloc[:, 2:8]

x_train,x_test, y_train, y_test = train_test_split(x,y, random_state=42, test_size = 0.25)

Count_vect = CountVectorizer(stop_words = 'english', ngram_range= (1,3), min_df=0.2)

Count_vect.fit_transform(x_train, y_train)
x_train_transform = Count_vect.transform(x_train)

x_test_tranform = Count_vect.transform(x_test)

col_names = list(y.columns)
cross_val_loss, cross_val_Score, accuracy = [], [], []

model = MultinomialNB()

for name in col_names:
    y_train_target = y_train[name]   
    y_test_target = y_test[name]
    cross_val_Score.append(np.mean(cross_val_score(model, x_train_transform, y_train_target, cv =10, scoring = 'accuracy')))
    cross_val_loss.append(np.mean(cross_val_score(model, x_train_transform, y_train_target, cv = 10, scoring= 'neg_log_loss')))
  
    model.fit(x_train_transform, y_train_target)
    
    prediction = model.predict(x_test_tranform)
    accuracy.append(accuracy_score(prediction,y_test_target))
    
    
