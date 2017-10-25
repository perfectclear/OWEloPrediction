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

train_df = pd.read_csv("OWDataTrain.csv")
test_df = pd.read_csv("OWDataTest.csv")
combine = [train_df, test_df]
heroes = ["ana","bastion","doomfist","dva","genji","hanzo","junkrat","lucio","mccree","mei","mercy","orisa","pharah","reaper","reinhardt","roadhog",
          "soldier76","sombra","symmetra","torbjorn","tracer","widowmaker","winston","zarya","zenyatta"]

def dfsplit(data, name):
    data = data.drop([title for title in data if 'heroes|stats|competitive|{NAME}'.format(NAME=name) not in title], axis=1)
    return data

  
for hero in heroes:
    globals()['%s'% hero + '_train_df'] = dfsplit(train_df, hero)
    globals()['%s'% hero + '_test_df'] = dfsplit(test_df, hero)
globals()['general_test_df'] = test_df.drop([title for title in test_df if 'heroes|stats|competitive|' in title], axis=1)
globals()['general_train_df'] = train_df.drop([title for title in train_df if 'heroes|stats|competitive|' in title], axis=1)

