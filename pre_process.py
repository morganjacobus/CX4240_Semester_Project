# -*- coding: utf-8 -*-
"""
Pre-Processing Python Script
"""
import pandas as pd
import random as rand

filename = "C:\Users\morga\Documents\Georgia Tech\Classes\CX 4240\Data\Train.csv"

n = sum(1 for line in open(filename)) - 1
s = 1000
skip = sorted(rand.sample(xrange(1,n+1),n-s))
df = pd.read_csv(filename, skiprows = skip)