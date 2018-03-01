# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""
import pandas as pd
import random as rand
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
filename = 'Test.csv'
n = sum(1 for line in open(filename))
s = 1000
skip = sorted(rand.sample(xrange(1,n+1),n-s))
df = pd.read_csv(filename,skiprows=skip)
txt_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),
                    ('RF', RandomForestClassifier())])
