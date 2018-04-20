# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""

"""Load Packages"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.ensemble import RakelD
from sklearn.pipeline import Pipeline
from skmultilearn.problem_transform import LabelPowerset
import re
from itertools import chain
from scipy import sparse
from scipy.optimize import curve_fit
import math
from collections import Counter
"""Import Training Data"""
filename = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Train.csv"
filename_test = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Test.csv"
#filename = "Train.csv"

"""Load Data Into Iterable Object"""
#Uncomment end portion if trying to iterate piecewise
#chunks = pd.read_csv(filename,chunksize = 10000,index_col=0)
chunks = pd.read_csv(filename,chunksize = 1000,index_col=0,iterator = True)
chunks_test = pd.read_csv(filename_test,chunksize = 1000, index_col = 0, iterator = True)
"""Retrieve Tags/y_train for Entire Dataset"""
#Use only if iterator = True is commented out
#labels = []
#i = 1
#index = []
#length = []
#for chunk in chunks:
#    for item in chunk["Tags"]:
#        labels.append(item.split())
#    index.append(i*10000)
    #length.append(len(MultiLabelBinarizer().fit_transform(labels)))
    #Uncomment if memory is good
#y_Train = MultiLabelBinarizer().fit_transform(labels)
#print(len(list(set(chain(*labels)))))    
#unique_labels = list(set(chain(*labels)))

def subsetData(chunk,num):
      subset = chunk.get_chunk(num)
      subset["Text"] = subset["Title"] + " " + subset["Body"]
      return subset
  
def yLabel(subset,binarizer):
    labels = []
    for item in subset["Tags"]:
        labels.append(item.split())
    subset["labels"] = labels
    mlb = binarizer
    y_train = mlb.fit_transform(subset["labels"])
    return y_train, mlb

def vectorizer(vecMethod,subset):
    vectorizer = vecMethod
    vectorizer.fit(subset)
    vector = vectorizer.transform(subset)
    X_vector = vector.toarray()
    return X_vector, vectorizer

def xTest(vectorizer,subset):
    vector_test = vectorizer.transform(subset)
    X_test = vector_test.toarray()
    return X_test

def output(outList,y_train):
    output_lables = []
    for x in outList:
        output_lables.append(" ".join(x))
    idLable = list(range(1,len(y_train)+1))
    d = {"Id":idLable,"Tags":output_lables}
    final_output = pd.DataFrame(d)
    return final_output

#----------------Load Data---------------------#
df = subsetData(chunks,1000)    
df_test = subsetData(chunks_test,1000)
#---------------------------------------------#

#---------------Create y Label Matrix---------#
y_train, mlb = yLabel(df,MultiLabelBinarizer())
#---------------------------------------------#

#--------Count Vecotrizer-----------------------#
"""Create X_train, y_train, and X_test"""
X_train, vect = vectorizer(CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS),df["Text"])
X_test = xTest(vect,df_test["Text"])

"""Random Forest for Count Vectorizer"""
base_classifier = RandomForestClassifier(random_state = 123)
problem_transform_classifier = LabelPowerset(classifier = base_classifier,
        require_dense = [False,False])
classifier = RakelD(problem_transform_classifier,len(y_train))
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
output_list = mlb.inverse_transform(predictions)
count_vec_output = output(output_list,y_train)

#------------------------------------------------#


#---------------------Tfidf Vectorizer--------------------#
X_train_tf_idf, tf_idf = vectorizer(TfidfVectorizer(),df["Text"])
X_test_tf_idf = xTest(tf_idf,df_test["Text"])

"""Random Forest for TfIdf Vectorizer"""
base_classifier = RandomForestClassifier()
problem_transform_classifier = LabelPowerset(classifier = base_classifier,
        require_dense = [False,False])
classifier = RakelD(problem_transform_classifier,len(y_train))
classifier.fit(X_train_tf_idf,y_train)
predictions = classifier.predict(X_test_tf_idf)
output_list = mlb.inverse_transform(predictions)
tf_idf_output = output(output_list,y_train)
#----------------------------------------------------------#


#--------------------Hashing Vectorizer--------------------#
X_train_hash , hashVec = vectorizer(HashingVectorizer(n_features = 200),df["Text"])
X_test_hash = xTest(hashVec,df_test["Text"])

"""Random Forest for Hashing Vectorizer"""
base_classifier = RandomForestClassifier()
problem_transform_classifier = LabelPowerset(classifier = base_classifier,
        require_dense = [False,False])
classifier = RakelD(problem_transform_classifier,len(y_train))
classifier.fit(X_train_hash,y_train)
predictions = classifier.predict(X_test_hash)
output_list = mlb.inverse_transform(predictions)
hash_output = output(output_list,y_train)







"""Count Vectorizer Histogram FOR SUBSET ONLY"""
c = vectorizer.vocabulary_.values()

n, bins, patches = plt.hist(c,10,facecolor="b",alpha = 0.5)
plt.xlabel("Word Counts")
plt.ylabel("Frequency")
plt.title("Histogram of Word Counts")
plt.show()

"""Tf-Idf Vectorizer to X_train FOR SUBSET ONLY"""
tf_idf_vectorizer = TfidfVectorizer()
tf_idf_vectorizer.fit(subset["Text"])
X_train = []
tf_idf_vector = tf_idf_vectorizer.transform(subset["Text"])
X_train = tf_idf_vector.toarray()

"""Tf-Idf Vectorizer Histogram FOR SUBSET ONLY"""
counts_tf_idf = tf_idf_vectorizer.vocabulary_.values()

n, bins, patches = plt.hist(counts_tf_idf,10000,facecolor="b",alpha = 0.5)
plt.xlabel("TF-IDF Counts")
plt.ylabel("Frequency")
plt.title("Histogram of TF-IDF")
plt.show()

"""Hashing Vectorizer FOR SUBSET ONLY"""
hash_vectorizer = HashingVectorizer(n_features = 200)
hash_vector = hash_vectorizer.transform(subset["Text"])
X_train = []
X_train = hash_vector.toarray()


"""Complexity Analysis FOR SUBSET ONLY"""
n = []
num_labels = []

for i in range(1000,1501000,1000):
    comp_data = chunks.get_chunk(i)
    comp_label = []
    for row in comp_data["Tags"]:
        comp_label.append(row.split())
    #comp_data["labels"] = comp_label
    #y_train_comp = MultiLabelBinarizer().fit_transform(comp_data["labels"])
    n.append(i)
    num_labels.append(len(list(set(chain(*comp_label)))))

    
line, = plt.plot(n,num_labels,lw =2)
plt.xlabel("No. of Observations")
plt.ylabel("No. of Labels")
plt.title("Plot of Label Complexity vs. Observation Count")
plt.show()

def func(x,a,b):
    return a*np.log(x) + b

params = curve_fit(func,n,num_labels)
print(params[0])

x_new = np.linspace(n[0],n[-1],1000)
y_new = func(x_new,params[0][0],params[0][1])
plt.plot(n,num_labels,"o",x_new,y_new)

n = [1] + n
num_labels = [5] + num_labels
np.polyfit(np.log(n),np.log(num_labels),1)
