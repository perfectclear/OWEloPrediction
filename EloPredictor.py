# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import math
import pylab as pl
import sklearn.cluster as cluster
import json
import ast
import collections

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = str(parent_key) + sep + str(k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

data = dict()
testmap = {}
with open('playerstats3.txt', 'r') as f:
    content = f.readlines()
    for i, line in enumerate(content[1:]):
        line = line[:-2]
        datum = ast.literal_eval(line)
        data.update({i: flatten(datum, sep = '|')})

df = pd.DataFrame.from_dict(data, orient='index')
df = df.dropna(axis=1, thresh = 20)

df.to_csv("OverwatchData.csv", index= False)
