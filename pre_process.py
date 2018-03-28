# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""
import pandas as pd
import random as rand
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import re
def cleanhtml(raw_html):
     cleanr = re.compile('<.*?>')
     cleantext = re.sub(cleanr, '', raw_html)
     return cleantext
filename = 'Train.csv'
chunks = pd.read_csv(filename,chunksize=1000,iterator=True)
subset= chunks.get_chunk(1000)
subset["Text"] = subset["Title"] + subset["Body"]
subset["Label"] = ""
for i in xrange(len(subset["Tags"])):
    subset["Label"][i] = subset["Tags"][i].split()
#txt_clf = Pipeline([('vect', CountVectorizer(stop_words = text.ENGLISH_STOP_WORDS)),
#                    ('tfidf', TfidfTransformer()),
#                    ('RF', RandomForestClassifier(n_estimators=500))])
count_vect = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
X_train_counts = count_vect.fit_transform(subset["Text"])
count_vect.vocabulary_.get(u'algorithm')
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
