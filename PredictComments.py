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


train_data = pd.read_csv(r'E:\GitHub\Comments_Prediction\data\train.csv')

train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\n", " ", x))
train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\"", "", x))
# train_data['comment_text'] = train_data['comment_text'].map(lambda x: re.sub("\n", " ", x))


# x = train_data['comment_text']
# y = train_data.iloc[:, 2:8]
