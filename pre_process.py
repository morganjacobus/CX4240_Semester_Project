# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""

"""Load Packages"""
import csv
import re
import math
import pandas as pd
import numpy as np
import scipy.sparse as sps
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
from itertools import chain
from scipy import sparse
from scipy.optimize import curve_fit
from collections import Counter

"""String for filename/path"""
filename = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Train.csv"
filename_test = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Test.csv"
#filename = "Train.csv"

"""Load Data Into Iterable Object"""
#Uncomment end portion if trying to iterate piecewise
#chunks = pd.read_csv(filename,chunksize = 10000,index_col=0)
#chunks = pd.read_csv(filename,chunksize = 1000,index_col=0,iterator = True)
#chunks_test = pd.read_csv(filename_test,chunksize = 1000, index_col = 0, iterator = True)


def subsetData(chunk,num):
      subset = chunk.get_chunk(num)
      subset["Text"] = subset["Title"] + " " + subset["Body"]
      return subset

def subsetData2(filename,num,column_start,column_end):
      subset = pd.read_csv(filename,nrows=num)
      #subset["Text"] = subset["Title"] + " " + subset["Body"]
      return subset.iloc[:,column_start:column_end]


def yLabel(subset,binarizer):
   labels = []
   for index,item in subset.iterrows():
       labels.append(item["Tags"].split())
   subset["Tags"] = labels
   mlb = binarizer
   mlb.fit(subset["Tags"])
   y_train = mlb.transform(subset["Tags"])
   return y_train, mlb
 
def vectorizer(vecMethod,subset):
    vectorizer = vecMethod
    vectorizer.fit(subset)
    vector = vectorizer.transform(subset)
    X_vector = vector
    return X_vector, vectorizer

def xTest(vectorizer,subset):
    vector_test = vectorizer.transform(subset)
    X_test = vector_test
    return X_test

def output(outList,test_set_length):
    output_lables = []
    for x in outList:
        output_lables.append(" ".join(x))
    idLable = list(range(1,test_set_length+1))
    d = {"Id":idLable,"Tags":output_lables}
    final_output = pd.DataFrame(d)
    return final_output





#----------------Load Data X_train/y_train---------------------#
#df = subsetData(chunks,1)
#df_test = subsetData(chunks_test,1)
#df_1, df_2 = df[:1000000], df[1000000:]
#df_test_1, df_test_2 = df_test[:1000000], df_test[1000000:]
"""------------------Count Vectorizer----------------------------"""
#---------------Load Data X_train/y_train/X_test---------#
#y_train, mlb = yLabel(df,MultiLabelBinarizer(sparse_output = True))
#X_train, vect = vectorizer(CountVectorizer(max_features = 40000,stop_words=text.ENGLISH_STOP_WORDS),df["Text"])
#X_test = xTest(vect,df_test["Text"])
#---------------------------------------------#

df = subsetData2(filename,4000000,2,4)
train_length = 4000000
df_test = subsetData2(filename_test,2000000,2,3)
test_set_length = len(df_test)
y_train, mlb = yLabel(df,MultiLabelBinarizer(sparse_output = True))
X_train, vect = vectorizer(CountVectorizer(max_features = 500000,stop_words=text.ENGLISH_STOP_WORDS),df["Body"])
X_test = xTest(vect,df_test["Body"])

#--------Random Forest Classifier-----------------------#
"""Random Forest for Count Vectorizer"""
base_classifier = RandomForestClassifier(random_state = 123)
problem_transform_classifier = LabelPowerset(classifier = base_classifier,
        require_dense = [False,False])
classifier = RakelD(problem_transform_classifier,train_length)
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
output_list = mlb.inverse_transform(predictions)
count_vec_output = output(output_list,test_set_length)
#------------------------------------------------#
"""----------------------------------------------------------------"""





#---------------------Tfidf Vectorizer--------------------#
class lazy_csv_reader(list):
    def __init__(self, csv_reader, pos):
        self.reader = csv_reader
        self.pos = pos
        
    def __iter__(self):
        r = self.reader.next()
        while r:
            yield r[self.pos]
            r = self.reader.next()

r = csv.reader(open(filename, 'rb'))
r.next() # get rid of the header
tfidf_vec = TfidfVectorizer()
tfidf = tfidf_vec.fit_transform(lazy_csv_reader(r, 2))
X_train = sparse.csr_matrix(tfidf)

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
tf_idf_output = output(output_list,X_test_tf_idf)
#----------------------------------------------------------#


#--------------------Hashing Vectorizer--------------------#
X_train_hash , hashVec = vectorizer(HashingVectorizer(n_features = 20),df["Body"])
y_train_hash, mlb = yLabel(df_1,MultiLabelBinarizer(sparse_output = True))
X_test_full = xTest(hashVec,df_test["Body"])

"""Random Forest for Hashing Vectorizer"""
base_classifier = RandomForestClassifier()
problem_transform_classifier = LabelPowerset(classifier = base_classifier,
        require_dense = [False,False])
classifier = RakelD(problem_transform_classifier,train_length)
classifier.fit(X_train_hash,y_train_hash)
predictions = classifier.predict(X_test_full)
output_list = mlb.inverse_transform(predictions)
hash_output = output(output_list,test_set_length)
hash_output.to_csv("hash_output.csv")




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
#    length.append(len(MultiLabelBinarizer().fit_transform(labels)))
#    Uncomment if memory is good
#y_Train = MultiLabelBinarizer().fit_transform(labels)
#print(len(list(set(chain(*labels)))))    
#unique_labels = list(set(chain(*labels)))




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

for i in range(10000,1510000,10000):
    comp_data = chunks.get_chunk(i)
    comp_label = []
    for row in comp_data["Tags"]:
        comp_label.append(row.split())
    #comp_data["labels"] = comp_label
    #y_train_comp = MultiLabelBinarizer().fit_transform(comp_data["labels"])
    n.append(i)
    num_labels.append(len(list(set(chain(*comp_label)))))

num_labels = []
for i in range(len(chunks)):
    for item in chunk["Tags"]:
        labels.append(item.split())

    
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
