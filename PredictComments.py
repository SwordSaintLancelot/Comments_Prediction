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


input_text = ["The plantations were filled with apple trees",
              "Deers and antelopes ate from the trees in the plantation ","Apple from the plantation trees was cheaper than others"]

count_vect = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.7)
count_vect.fit_transform(input_text)
feature_list = count_vect.get_feature_names()