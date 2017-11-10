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

df = pd.read_csv('OverwatchData.csv')
train_df = df[:len(df)*4//5]
test_df = df[len(df)*4//5:]
del(df)

combine = [train_df, test_df]
heroes = ["ana","bastion","doomfist","dva","genji","hanzo","junkrat","lucio","mccree","mei","mercy","orisa","pharah","reaper","reinhardt","roadhog",
          "soldier76","sombra","symmetra","torbjorn","tracer","widowmaker","winston","zarya","zenyatta"]
droplist = ['rank', 'avatar', 'tier', 'rolling_average_stats', 'kill_streak_best', 'guid', 'final_blow_most_in_game', 'solo_kill_most_in_game', 'solo_kill_average', 'final_blow_average']
droplist2= ['solo_kill', 'final_blow']

for dataset in combine:
    dataset.insert(0, 'CompRank', dataset['stats|competitive|overall_stats|comprank'])
for title in droplist:
    train_df = train_df.drop([name for name in train_df if title in name], axis=1)
    test_df = test_df.drop([name for name in test_df if title in name], axis=1)
for hero in heroes:
    for title in droplist2:
            train_df = train_df.drop(['heroes|stats|competitive|{NAME}|general_stats|{TITLE}'.format(NAME= hero, TITLE= title)], axis=1)
            test_df = test_df.drop(['heroes|stats|competitive|{NAME}|general_stats|{TITLE}'.format(NAME= hero, TITLE= title)], axis=1)
train_df.to_csv("OWDataTrain5.csv", index = False)
test_df.to_csv("OWDataTest5.csv", index = False)