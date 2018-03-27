# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
filename = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Train.csv"


chunks = pd.read_csv(filename,chunksize = 1000,iterator = True,index_col = 0)

subset = chunks.get_chunk(1001)

subset["Text"] = subset["Title"] + " " + subset["Body"]

labels = []

for item in subset["Tags"]:
    labels.append(item.split())

subset["labels"] = labels
y_train = MultiLabelBinarizer().fit_transform(subset["labels"])


vectorizer = CountVectorizer()

vectorizer.fit(subset["Text"])

print(vectorizer.vocabulary_)

X_train = []

vector = vectorizer.transform(subset["Text"])

X_train = vector.toarray()

