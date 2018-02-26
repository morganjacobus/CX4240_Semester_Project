# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""
import pandas as pd
import random as rand
from sklearn.feature_extraction.text import CountVectorizer

filename = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Train.csv"

n = sum(1 for line in open(filename)) - 1
s = 1000
skip = sorted(rand.sample(xrange(1,n+1),n-s))
df = pd.read_csv(filename, skiprows = skip)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(df['Body'])
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')
