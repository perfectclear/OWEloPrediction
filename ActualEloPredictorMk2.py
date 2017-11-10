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

train_df = pd.read_csv("OWDataTrain5.csv")
test_df = pd.read_csv("OWDataTest5.csv")
combine = [train_df, test_df]
heroes = ["ana","bastion","doomfist","dva","genji","hanzo","junkrat","lucio","mccree","mei","mercy","orisa","pharah","reaper","reinhardt","roadhog",
          "soldier76","sombra","symmetra","torbjorn","tracer","widowmaker","winston","zarya","zenyatta"]
#splitting into dataframes for each hero, and general data
def dfsplit(data, name):
    data = data.drop([title for title in data if ('heroes|stats|competitive|{NAME}'.format(NAME=name) not in title and 'CompRank' not in title)], axis=1)
    return data

d = {}  
for hero in heroes:
    d['%s'% hero + '_train_df'] = dfsplit(train_df, hero)
    d['%s'% hero + '_test_df'] = dfsplit(test_df, hero)
d['general_test_df'] = test_df.drop([title for title in test_df if 'heroes|stats|competitive|' in title], axis=1)
d['general_train_df'] = train_df.drop([title for title in train_df if 'heroes|stats|competitive|' in title], axis=1)

#change CompRank to Bands of 250 width
for df in d:
    d[df].loc[ d[df]['CompRank'] <= 250, 'CompRank'] = 0
    d[df].loc[(d[df]['CompRank'] > 250) & (d[df]['CompRank'] <= 500), 'CompRank'] = 1
    d[df].loc[(d[df]['CompRank'] > 500) & (d[df]['CompRank'] <= 750), 'CompRank'] = 2
    d[df].loc[(d[df]['CompRank'] > 750) & (d[df]['CompRank'] <= 1000), 'CompRank'] = 3
    d[df].loc[(d[df]['CompRank'] > 1000) & (d[df]['CompRank'] <= 1250), 'CompRank'] = 4
    d[df].loc[(d[df]['CompRank'] > 1250) & (d[df]['CompRank'] <= 1500), 'CompRank'] = 5
    d[df].loc[(d[df]['CompRank'] > 1500) & (d[df]['CompRank'] <= 1750), 'CompRank'] = 6
    d[df].loc[(d[df]['CompRank'] > 1750) & (d[df]['CompRank'] <= 2000), 'CompRank'] = 7
    d[df].loc[(d[df]['CompRank'] > 2000) & (d[df]['CompRank'] <= 2250), 'CompRank'] = 8
    d[df].loc[(d[df]['CompRank'] > 2250) & (d[df]['CompRank'] <= 2500), 'CompRank'] = 9
    d[df].loc[(d[df]['CompRank'] > 2500) & (d[df]['CompRank'] <= 2750), 'CompRank'] = 10
    d[df].loc[(d[df]['CompRank'] > 2750) & (d[df]['CompRank'] <= 3000), 'CompRank'] = 11
    d[df].loc[(d[df]['CompRank'] > 3000) & (d[df]['CompRank'] <= 3250), 'CompRank'] = 12
    d[df].loc[(d[df]['CompRank'] > 3250) & (d[df]['CompRank'] <= 3500), 'CompRank'] = 13
    d[df].loc[(d[df]['CompRank'] > 3500) & (d[df]['CompRank'] <= 3750), 'CompRank'] = 14
    d[df].loc[(d[df]['CompRank'] > 3750) & (d[df]['CompRank'] <= 4000), 'CompRank'] = 15
    d[df].loc[(d[df]['CompRank'] > 4000) & (d[df]['CompRank'] <= 4250), 'CompRank'] = 16
    d[df].loc[(d[df]['CompRank'] > 4250) & (d[df]['CompRank'] <= 4500), 'CompRank'] = 17
    d[df].loc[(d[df]['CompRank'] > 4500) & (d[df]['CompRank'] <= 4750), 'CompRank'] = 18
    d[df].loc[(d[df]['CompRank'] > 4750), 'CompRank'] = 19


###########################################GENERAL STATS PREDICTOR####################################################

#lists for dropping unusable data
gen_droplist = ['stats|competitive|average_stats']

gen_droplist_exact = ['stats|competitive|game_stats|weapon_accuracy', 'stats|competitive|game_stats|melee_percentage_of_final_blows',
                   'stats|competitive|game_stats|of_teams_hero_damage', 'stats|competitive|game_stats|of_teams_damage',
                   'stats|competitive|game_stats|damage_blocked', 'stats|competitive|game_stats|environmental_kills_most_in_game',
                   'stats|competitive|game_stats|melee_final_blows_most_in_game', 'stats|competitive|game_stats|recon_assists_most_in_game',
                   'stats|competitive|game_stats|shield_generators_destroyed_most_in_game', 'stats|competitive|game_stats|teleporter_pads_destroyed_most_in_game',
                   'stats|competitive|game_stats|turrets_destroyed_most_in_game']

gen_combinelist = [['card', 'cards'], ['defensive_assist', 'defensive_assists'],
                ['defensive_assist_most_in_game', 'defensive_assists_most_in_game'],
                ['environmental_death','environmental_deaths'], ['environmental_kill', 'environmental_kills'],
                ['environmental_kill_most_in_game', 'environmental_kills_most_in_game'],
                ['melee_final_blow', 'melee_final_blows'], ['multikill','multikills'],
                ['offensive_assist', 'offensive_assists'], ['recon_assist', 'recon_assists'],
                ['offensive_assist_most_in_game', 'offensive_assists_most_in_game'],
                ['recon_assist_most_in_game', 'recon_assists_most_in_game'],
                ['shield_generator_destroyed', 'shield_generators_destroyed'],
                ['shield_generator_destroyed_most_in_game', 'shield_generators_destroyed_most_in_game'],
                ['solo_kill', 'solo_kills'], ['teleporter_pad_destroyed', 'teleporter_pads_destroyed'],
                ['teleporter_pad_destroyed_most_in_game', 'teleporter_pads_destroyed_most_in_game'],
                ['turret_destroyed', 'turrets_destroyed'], ['turret_destroyed_most_in_game', 'turrets_destroyed_most_in_game']]


#combining data that was input improperly to fill gaps in data
def dfcombine(data, combinelist):
 for item in combinelist:
     data['stats|competitive|game_stats|'+item[1]] = data['stats|competitive|game_stats|'+item[1]].combine_first(data['stats|competitive|game_stats|'+item[0]])
     data = data.drop('stats|competitive|game_stats|'+item[0], axis=1)
 return data

d['general_train_df'] = dfcombine(d['general_train_df'], gen_combinelist)
d['general_test_df'] = dfcombine(d['general_test_df'], gen_combinelist)


#predicting and filling in missing data for a few cases
"""if barrier damage is not null, sum the barrier damages and sum the all damage dones. ratio = barrier damage/all damage. barrier damage= ratio*all damage"""
def predictandfill(dataset, fillthis, predictby):
 fillthis_sum = 0
 damage_sum = 0
 for i, player in enumerate(dataset.transpose()):
     if dataset[fillthis][player] and not np.isnan(dataset[fillthis][player]):
#            print('dataset[fillthis][player]: {}'.format(dataset[fillthis][player]))
         fillthis_sum += dataset[fillthis][player]
#            print('fillthis_sum: {}'.format(fillthis_sum))
         damage_sum += dataset[predictby][player]
#            print('damage_sum: {}'.format(damage_sum))
 ratio = fillthis_sum / damage_sum
#    print(ratio)
 for i, player in enumerate(dataset.transpose()):
     if not dataset[fillthis][player] or np.isnan(dataset[fillthis][player]):
         dataset.loc[player, fillthis] = dataset[predictby][player] * ratio
 return dataset

d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|barrier_damage_done', 'stats|competitive|game_stats|all_damage_done')
d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|hero_damage_done', 'stats|competitive|game_stats|all_damage_done')
d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|hero_damage_done_most_in_game', 'stats|competitive|game_stats|all_damage_done_most_in_game')
d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|barrier_damage_done', 'stats|competitive|game_stats|all_damage_done')
d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|hero_damage_done', 'stats|competitive|game_stats|all_damage_done')
d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|hero_damage_done_most_in_game', 'stats|competitive|game_stats|all_damage_done_most_in_game')


#dropping unusable data
for title in gen_droplist:
 d['general_train_df'] = d['general_train_df'].drop([name for name in d['general_train_df'] if title in name], axis=1)
 d['general_test_df'] = d['general_test_df'].drop([name for name in d['general_test_df'] if title in name], axis=1)
for title in gen_droplist_exact:
 d['general_train_df'] = d['general_train_df'].drop([name for name in d['general_train_df'] if title in name], axis=1)
 d['general_test_df'] = d['general_test_df'].drop([name for name in d['general_test_df'] if title in name], axis=1)

#fill the remaining nans with 0
d['general_train_df'] = d['general_train_df'].fillna(0)
d['general_test_df'] = d['general_test_df'].fillna(0)

#fix level to be actual in game level
d['general_train_df']['stats|competitive|overall_stats|level'] = d['general_train_df']['stats|competitive|overall_stats|level']+100*d['general_train_df']['stats|competitive|overall_stats|prestige']
d['general_train_df'] = d['general_train_df'].drop(['stats|competitive|overall_stats|prestige'], axis=1)
d['general_test_df']['stats|competitive|overall_stats|level'] = d['general_test_df']['stats|competitive|overall_stats|level']+100*d['general_test_df']['stats|competitive|overall_stats|prestige']
d['general_test_df'] = d['general_test_df'].drop(['stats|competitive|overall_stats|prestige'], axis=1)

#change many stats to be on a per game basis
titles_to_change_train = [title for title in d['general_train_df'] if ('stats|competitive|game_stats' in title and 'games_played' not in title and 'kpd' not in title and 'most' not in title and 'CompRank' not in title)]
titles_to_change_test = [title for title in d['general_test_df'] if ('stats|competitive|game_stats' in title and 'games_played' not in title and 'kpd' not in title and 'most' not in title and 'CompRank' not in title)]
for title in titles_to_change_train:
 d['general_train_df']['%s'% title + '_per_game'] = d['general_train_df'][title]/d['general_train_df']['stats|competitive|game_stats|games_played']
for title in titles_to_change_test:
 d['general_test_df']['%s'% title + '_per_game'] = d['general_test_df'][title]/d['general_test_df']['stats|competitive|game_stats|games_played']
d['general_test_df'] = d['general_test_df'].drop(titles_to_change_test, axis=1)
d['general_train_df'] = d['general_train_df'].drop(titles_to_change_train, axis=1)

#change hero play time to be percentage of play time

total_hero_playtime_train = sum([d['general_train_df'][title] for title in d['general_train_df'] if 'heroes|playtime|competitive' in title])
for title in [title for title in d['general_train_df'] if 'heroes|playtime|competitive' in title]:
 d['general_train_df']['%s'% title + '_percentage'] = d['general_train_df'][title]/total_hero_playtime_train
 d['general_train_df'] = d['general_train_df'].drop([title], axis=1)

total_hero_playtime_test = sum([d['general_test_df'][title] for title in d['general_test_df'] if 'heroes|playtime|competitive' in title])
for title in [title for title in d['general_test_df'] if 'heroes|playtime|competitive' in title]:
 d['general_test_df']['%s'% title + '_percentage'] = d['general_test_df'][title]/total_hero_playtime_test
 d['general_test_df'] = d['general_test_df'].drop([title], axis=1)



##lets me see how much data is missing
#def missing_values_table(df):
#        mis_val = df.isnull().sum()
#        mis_val_percent = 100 * df.isnull().sum()/len(df)
#        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
#        mis_val_table_ren_columns = mis_val_table.rename(
#        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
#        return mis_val_table_ren_columns
#d2 = [missing_values_table(d['general_train_df']).sort_index(), missing_values_table(d['general_test_df']).sort_index()]



###playground! try dropping these?
#try_dropping = ['stats|competitive|game_stats|all_damage_done_most_in_game',
#                 'stats|competitive|overall_stats|win_rate',
#                 'stats|competitive|overall_stats|games',
#                 'stats|competitive|game_stats|eliminations_per_game',
#                 'stats|competitive|game_stats|final_blows_per_game',
#                 'stats|competitive|game_stats|medals_silver_per_game',
#                 'stats|competitive|game_stats|medals_bronze_per_game',
#                 'stats|competitive|game_stats|medals_per_game',
#                 'stats|competitive|game_stats|all_damage_done_per_game',
#                 'stats|competitive|game_stats|melee_final_blows_per_game',
#                 'stats|competitive|game_stats|multikill_best_per_game',
#                 'stats|competitive|game_stats|environmental_kills_per_game',
#                 'stats|competitive|game_stats|environmental_deaths_per_game',
#                 'stats|competitive|game_stats|defensive_assists_most_in_game', 
#                 'stats|competitive|game_stats|eliminations_most_in_game', 
#                 'stats|competitive|game_stats|final_blows_most_in_game', 
#                 'stats|competitive|game_stats|games_lost_per_game', 
#                 'stats|competitive|game_stats|games_tied_per_game', 
#                 'stats|competitive|game_stats|games_won_per_game', 
#                 'stats|competitive|game_stats|healing_done_most_in_game', 
#                 'stats|competitive|game_stats|hero_damage_done_most_in_game', 
#                 'stats|competitive|game_stats|objective_kills_most_in_game', 
#                 'stats|competitive|game_stats|objective_time_most_in_game', 
#                 'stats|competitive|game_stats|offensive_assists_most_in_game', 
#                 'stats|competitive|game_stats|solo_kills_most_in_game', 
#                 'stats|competitive|game_stats|time_spent_on_fire_most_in_game']
#
#playtimes = [feature for feature in d['general_train_df'] if 'playtime' in feature]
#for feature in playtimes:
#    try_dropping.append(feature)
#    
#for feature in try_dropping:
#    d['general_train_df'] = d['general_train_df'].drop([feature], axis=1)
#    d['general_test_df'] = d['general_test_df'].drop([feature], axis=1)


#maybe_drop = [feature for feature in d['general_train_df'] if ('CompRank' not in feature and 'overall_stats' not in feature and 'cards' not in feature and 'games' not in feature and 'medals' not in feature and 'deaths' not in feature)]
#
#for feature in maybe_drop:
#    d['general_train_df'] = d['general_train_df'].drop([feature], axis=1)
#    d['general_test_df'] = d['general_test_df'].drop([feature], axis=1)

### what if I just ran it without any further work?


X_gen_train = d['general_train_df'].drop("CompRank", axis=1)
Y_gen_train = d['general_train_df']["CompRank"]
X_gen_test  = d['general_test_df'].drop("CompRank", axis=1).copy()
Y_gen_test = d['general_test_df']["CompRank"]
# X_gen_train.shape, Y_gen_train.shape, X_gen_test.shape

#making sure accuracy is a useable thing
def accurate_to(data,thresh):
 matches = [i for i, j in zip(data, Y_gen_test) if abs(i-j)<=thresh]
 accuracy = len(matches)/len(Y_gen_test)
 return accuracy

# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_gen_train, Y_gen_train)
Y_gen_pred = logreg.predict(X_gen_test)
acc_log = round(logreg.score(X_gen_train, Y_gen_train) * 100, 2)
# print('acc_log'+str(acc_log))

coeff_df = pd.DataFrame(d['general_train_df'].columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

print("coeff_df.sort_values(by='Correlation', ascending=False)"+str(coeff_df.sort_values(by='Correlation', ascending=False)))

# Support Vector Machines

svc = SVC()
svc.fit(X_gen_train, Y_gen_train)
Y_gen_pred = svc.predict(X_gen_test)
acc_svc = round(svc.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_svc'+str(acc_svc))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_svc1'+str(actual_accuracy))


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_gen_train, Y_gen_train)
Y_gen_pred = knn.predict(X_gen_test)
acc_knn = round(knn.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_knn'+str(acc_knn))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_knn'+str(actual_accuracy))

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_gen_train, Y_gen_train)
Y_gen_pred = gaussian.predict(X_gen_test)
acc_gaussian = round(gaussian.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_gaussian'+str(acc_gaussian))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_gauss'+str(actual_accuracy))

# Perceptron

perceptron = Perceptron()
perceptron.fit(X_gen_train, Y_gen_train)
Y_gen_pred = perceptron.predict(X_gen_test)
acc_perceptron = round(perceptron.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_perceptron'+str(acc_perceptron))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_perceptron'+str(actual_accuracy))

# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_gen_train, Y_gen_train)
Y_gen_pred = linear_svc.predict(X_gen_test)
acc_linear_svc = round(linear_svc.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_linear_svc'+str(acc_linear_svc))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_svc'+str(actual_accuracy))

# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_gen_train, Y_gen_train)
Y_gen_pred = sgd.predict(X_gen_test)
acc_sgd = round(sgd.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_sgd'+str(acc_sgd))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_sgd'+str(actual_accuracy))

# AdaBoost

Ada_Boost = AdaBoostClassifier(n_estimators=100)
Ada_Boost.fit(X_gen_train, Y_gen_train)
Y_gen_pred = Ada_Boost.predict(X_gen_test)
Ada_Boost.score(X_gen_train, Y_gen_train)
acc_Ada_Boost = round(Ada_Boost.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_Ada_Boost'+str(acc_Ada_Boost))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_Ada_Boost'+str(actual_accuracy))

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_gen_train, Y_gen_train)
Y_gen_pred = decision_tree.predict(X_gen_test)
acc_decision_tree = round(decision_tree.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_decision_tree'+str(acc_decision_tree))
actual_accuracy = accurate_to(Y_gen_pred, 0)
print('actual_accuracy_tree'+str(actual_accuracy))


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_gen_train, Y_gen_train)
Y_gen_predz = random_forest.predict(X_gen_test)
random_forest.score(X_gen_train, Y_gen_train)
acc_random_forest = round(random_forest.score(X_gen_train, Y_gen_train) * 100, 2)
print('acc_random_forest'+str(acc_random_forest))
actual_accuracy = accurate_to(Y_gen_predz, 0)
print('actual_accuracy_forrest'+str(actual_accuracy))

#models = pd.DataFrame({
#    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
#              'Random Forest', 'Naive Bayes', 'Perceptron',
#              'Stochastic Gradient Decent', 'Linear SVC',
#              'Decision Tree', 'AdaBoost'],
#    'Score': [acc_svc, acc_knn, acc_log,
#              acc_random_forest, acc_gaussian, acc_perceptron,
#              acc_sgd, acc_linear_svc, acc_decision_tree, acc_Ada_Boost]})
#print("models.sort_values(by='Score', ascending=False)"+str(models.sort_values(by='Score', ascending=False)))

#submission3 = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
#        "Survived": Y_gen_pred
#    })
#submission3

####################################################ANA_PREDICTOR#########################################################
## Ana players only please! Others need not apply!
#d['ana_train_df'] = d['ana_train_df'].dropna(axis=0, subset = [title for title in d['ana_train_df'] if 'CompRank' not in title], how='all')
#d['ana_test_df'] = d['ana_test_df'].dropna(axis=0, subset = [title for title in d['ana_test_df'] if 'CompRank' not in title], how='all')
#d['ana_train_df'] = d['ana_train_df'].drop(d['ana_train_df'][d['general_train_df']['heroes|playtime|competitive|ana_percentage'] < 0.01].index)
#d['ana_test_df'] = d['ana_test_df'].drop(d['ana_test_df'][d['general_test_df']['heroes|playtime|competitive|ana_percentage'] < 0.01].index)
#
###lists for dropping unusable data
##gen_droplist = ['stats|competitive|average_stats']
##
##gen_droplist_exact = ['stats|competitive|game_stats|weapon_accuracy', 'stats|competitive|game_stats|melee_percentage_of_final_blows',
##                      'stats|competitive|game_stats|of_teams_hero_damage', 'stats|competitive|game_stats|of_teams_damage',
##                      'stats|competitive|game_stats|damage_blocked', 'stats|competitive|game_stats|environmental_kills_most_in_game',
##                      'stats|competitive|game_stats|melee_final_blows_most_in_game', 'stats|competitive|game_stats|recon_assists_most_in_game',
##                      'stats|competitive|game_stats|shield_generators_destroyed_most_in_game', 'stats|competitive|game_stats|teleporter_pads_destroyed_most_in_game',
##                      'stats|competitive|game_stats|turrets_destroyed_most_in_game']
##
##gen_combinelist = [['card', 'cards'], ['defensive_assist', 'defensive_assists'],
##                   ['defensive_assist_most_in_game', 'defensive_assists_most_in_game'],
##                   ['environmental_death','environmental_deaths'], ['environmental_kill', 'environmental_kills'],
##                   ['environmental_kill_most_in_game', 'environmental_kills_most_in_game'],
##                   ['melee_final_blow', 'melee_final_blows'], ['multikill','multikills'],
##                   ['offensive_assist', 'offensive_assists'], ['recon_assist', 'recon_assists'],
##                   ['offensive_assist_most_in_game', 'offensive_assists_most_in_game'],
##                   ['recon_assist_most_in_game', 'recon_assists_most_in_game'],
##                   ['shield_generator_destroyed', 'shield_generators_destroyed'],
##                   ['shield_generator_destroyed_most_in_game', 'shield_generators_destroyed_most_in_game'],
##                   ['solo_kill', 'solo_kills'], ['teleporter_pad_destroyed', 'teleporter_pads_destroyed'],
##                   ['teleporter_pad_destroyed_most_in_game', 'teleporter_pads_destroyed_most_in_game'],
##                   ['turret_destroyed', 'turrets_destroyed'], ['turret_destroyed_most_in_game', 'turrets_destroyed_most_in_game']]
##
##
###combining data that was input improperly to fill gaps in data
##def dfcombine(data, combinelist):
##    for item in combinelist:
##        data['stats|competitive|game_stats|'+item[1]] = data['stats|competitive|game_stats|'+item[1]].combine_first(data['stats|competitive|game_stats|'+item[0]])
##        data = data.drop('stats|competitive|game_stats|'+item[0], axis=1)
##    return data
##
##d['general_train_df'] = dfcombine(d['general_train_df'], gen_combinelist)
##d['general_test_df'] = dfcombine(d['general_test_df'], gen_combinelist)
##
##
###predicting and filling in missing data for a few cases
##"""if barrier damage is not null, sum the barrier damages and sum the all damage dones. ratio = barrier damage/all damage. barrier damage= ratio*all damage"""
##def predictandfill(dataset, fillthis, predictby):
##    fillthis_sum = 0
##    damage_sum = 0
##    for i, player in enumerate(dataset.transpose()):
##        if dataset[fillthis][player] and not np.isnan(dataset[fillthis][player]):
###            print('dataset[fillthis][player]: {}'.format(dataset[fillthis][player]))
##            fillthis_sum += dataset[fillthis][player]
###            print('fillthis_sum: {}'.format(fillthis_sum))
##            damage_sum += dataset[predictby][player]
###            print('damage_sum: {}'.format(damage_sum))
##    ratio = fillthis_sum / damage_sum
###    print(ratio)
##    for i, player in enumerate(dataset.transpose()):
##        if not dataset[fillthis][player] or np.isnan(dataset[fillthis][player]):
##            dataset.loc[player, fillthis] = dataset[predictby][player] * ratio
##    return dataset
##
##d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|barrier_damage_done', 'stats|competitive|game_stats|all_damage_done')
##d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|hero_damage_done', 'stats|competitive|game_stats|all_damage_done')
##d['general_train_df'] = predictandfill(d['general_train_df'], 'stats|competitive|game_stats|hero_damage_done_most_in_game', 'stats|competitive|game_stats|all_damage_done_most_in_game')
##d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|barrier_damage_done', 'stats|competitive|game_stats|all_damage_done')
##d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|hero_damage_done', 'stats|competitive|game_stats|all_damage_done')
##d['general_test_df'] = predictandfill(d['general_test_df'], 'stats|competitive|game_stats|hero_damage_done_most_in_game', 'stats|competitive|game_stats|all_damage_done_most_in_game')
##
##
###dropping unusable data
##for title in gen_droplist:
##    d['general_train_df'] = d['general_train_df'].drop([name for name in d['general_train_df'] if title in name], axis=1)
##    d['general_test_df'] = d['general_test_df'].drop([name for name in d['general_test_df'] if title in name], axis=1)
##for title in gen_droplist_exact:
##    d['general_train_df'] = d['general_train_df'].drop([name for name in d['general_train_df'] if title in name], axis=1)
##    d['general_test_df'] = d['general_test_df'].drop([name for name in d['general_test_df'] if title in name], axis=1)
##
###fill the remaining nans with 0
##d['general_train_df'] = d['general_train_df'].fillna(0)
##d['general_test_df'] = d['general_test_df'].fillna(0)
##
###fix level to be actual in game level
##d['general_train_df']['stats|competitive|overall_stats|level'] = d['general_train_df']['stats|competitive|overall_stats|level']+100*d['general_train_df']['stats|competitive|overall_stats|prestige']
##d['general_train_df'] = d['general_train_df'].drop(['stats|competitive|overall_stats|prestige'], axis=1)
##d['general_test_df']['stats|competitive|overall_stats|level'] = d['general_test_df']['stats|competitive|overall_stats|level']+100*d['general_test_df']['stats|competitive|overall_stats|prestige']
##d['general_test_df'] = d['general_test_df'].drop(['stats|competitive|overall_stats|prestige'], axis=1)
##
###change many stats to be on a per game basis
##titles_to_change_train = [title for title in d['general_train_df'] if ('stats|competitive|game_stats' in title and 'games_played' not in title and 'kpd' not in title and 'most' not in title and 'CompRank' not in title)]
##titles_to_change_test = [title for title in d['general_test_df'] if ('stats|competitive|game_stats' in title and 'games_played' not in title and 'kpd' not in title and 'most' not in title and 'CompRank' not in title)]
##for title in titles_to_change_train:
##    d['general_train_df']['%s'% title + '_per_game'] = d['general_train_df'][title]/d['general_train_df']['stats|competitive|game_stats|games_played']
##for title in titles_to_change_test:
##    d['general_test_df']['%s'% title + '_per_game'] = d['general_test_df'][title]/d['general_test_df']['stats|competitive|game_stats|games_played']
##d['general_test_df'] = d['general_test_df'].drop(titles_to_change_test, axis=1)
##d['general_train_df'] = d['general_train_df'].drop(titles_to_change_train, axis=1)
##
###change hero play time to be percentage of play time
##
##total_hero_playtime_train = sum([d['general_train_df'][title] for title in d['general_train_df'] if 'heroes|playtime|competitive' in title])
##for title in [title for title in d['general_train_df'] if 'heroes|playtime|competitive' in title]:
##    d['general_train_df']['%s'% title + '_percentage'] = d['general_train_df'][title]/total_hero_playtime_train
##    d['general_train_df'] = d['general_train_df'].drop([title], axis=1)
##
##total_hero_playtime_test = sum([d['general_test_df'][title] for title in d['general_test_df'] if 'heroes|playtime|competitive' in title])
##for title in [title for title in d['general_test_df'] if 'heroes|playtime|competitive' in title]:
##    d['general_test_df']['%s'% title + '_percentage'] = d['general_test_df'][title]/total_hero_playtime_test
##    d['general_test_df'] = d['general_test_df'].drop([title], axis=1)
##
###change CompRank to Bands of 250 width
##d['general_train_df'].loc[ d['general_train_df']['CompRank'] <= 250, 'CompRank'] = 0
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 250) & (d['general_train_df']['CompRank'] <= 500), 'CompRank'] = 1
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 500) & (d['general_train_df']['CompRank'] <= 750), 'CompRank'] = 2
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 750) & (d['general_train_df']['CompRank'] <= 1000), 'CompRank'] = 3
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 1000) & (d['general_train_df']['CompRank'] <= 1250), 'CompRank'] = 4
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 1250) & (d['general_train_df']['CompRank'] <= 1500), 'CompRank'] = 5
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 1500) & (d['general_train_df']['CompRank'] <= 1750), 'CompRank'] = 6
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 1750) & (d['general_train_df']['CompRank'] <= 2000), 'CompRank'] = 7
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 2000) & (d['general_train_df']['CompRank'] <= 2250), 'CompRank'] = 8
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 2250) & (d['general_train_df']['CompRank'] <= 2500), 'CompRank'] = 9
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 2500) & (d['general_train_df']['CompRank'] <= 2750), 'CompRank'] = 10
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 2750) & (d['general_train_df']['CompRank'] <= 3000), 'CompRank'] = 11
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 3000) & (d['general_train_df']['CompRank'] <= 3250), 'CompRank'] = 12
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 3250) & (d['general_train_df']['CompRank'] <= 3500), 'CompRank'] = 13
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 3500) & (d['general_train_df']['CompRank'] <= 3750), 'CompRank'] = 14
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 3750) & (d['general_train_df']['CompRank'] <= 4000), 'CompRank'] = 15
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 4000) & (d['general_train_df']['CompRank'] <= 4250), 'CompRank'] = 16
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 4250) & (d['general_train_df']['CompRank'] <= 4500), 'CompRank'] = 17
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 4500) & (d['general_train_df']['CompRank'] <= 4750), 'CompRank'] = 18
##d['general_train_df'].loc[(d['general_train_df']['CompRank'] > 4750), 'CompRank'] = 19
##
##d['general_test_df'].loc[ d['general_test_df']['CompRank'] <= 250, 'CompRank'] = 0
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 250) & (d['general_test_df']['CompRank'] <= 500), 'CompRank'] = 1
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 500) & (d['general_test_df']['CompRank'] <= 750), 'CompRank'] = 2
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 750) & (d['general_test_df']['CompRank'] <= 1000), 'CompRank'] = 3
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 1000) & (d['general_test_df']['CompRank'] <= 1250), 'CompRank'] = 4
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 1250) & (d['general_test_df']['CompRank'] <= 1500), 'CompRank'] = 5
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 1500) & (d['general_test_df']['CompRank'] <= 1750), 'CompRank'] = 6
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 1750) & (d['general_test_df']['CompRank'] <= 2000), 'CompRank'] = 7
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 2000) & (d['general_test_df']['CompRank'] <= 2250), 'CompRank'] = 8
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 2250) & (d['general_test_df']['CompRank'] <= 2500), 'CompRank'] = 9
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 2500) & (d['general_test_df']['CompRank'] <= 2750), 'CompRank'] = 10
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 2750) & (d['general_test_df']['CompRank'] <= 3000), 'CompRank'] = 11
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 3000) & (d['general_test_df']['CompRank'] <= 3250), 'CompRank'] = 12
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 3250) & (d['general_test_df']['CompRank'] <= 3500), 'CompRank'] = 13
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 3500) & (d['general_test_df']['CompRank'] <= 3750), 'CompRank'] = 14
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 3750) & (d['general_test_df']['CompRank'] <= 4000), 'CompRank'] = 15
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 4000) & (d['general_test_df']['CompRank'] <= 4250), 'CompRank'] = 16
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 4250) & (d['general_test_df']['CompRank'] <= 4500), 'CompRank'] = 17
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 4500) & (d['general_test_df']['CompRank'] <= 4750), 'CompRank'] = 18
##d['general_test_df'].loc[(d['general_test_df']['CompRank'] > 4750), 'CompRank'] = 19
##
####lets me see how much data is missing
###def missing_values_table(df):
###        mis_val = df.isnull().sum()
###        mis_val_percent = 100 * df.isnull().sum()/len(df)
###        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
###        mis_val_table_ren_columns = mis_val_table.rename(
###        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
###        return mis_val_table_ren_columns
###d2 = [missing_values_table(d['general_train_df']).sort_index(), missing_values_table(d['general_test_df']).sort_index()]
##
##
##
##
##
##
##
##
##
##### what if I just ran it without any further work?
##
##
##X_gen_train = d['general_train_df'].drop("CompRank", axis=1)
##Y_gen_train = d['general_train_df']["CompRank"]
##X_gen_test  = d['general_test_df'].drop("CompRank", axis=1).copy()
##Y_gen_test = d['general_test_df']["CompRank"]
### X_gen_train.shape, Y_gen_train.shape, X_gen_test.shape
##
###making sure accuracy is a useable thing
##def accurate_to(data,thresh):
##    matches = [i for i, j in zip(data, Y_gen_test) if abs(i-j)<=thresh]
##    accuracy = len(matches)/len(Y_gen_test)
##    return accuracy
##
### Logistic Regression
##
##logreg = LogisticRegression()
##logreg.fit(X_gen_train, Y_gen_train)
##Y_gen_pred = logreg.predict(X_gen_test)
##acc_log = round(logreg.score(X_gen_train, Y_gen_train) * 100, 2)
### print('acc_log'+str(acc_log))
##
##coeff_df = pd.DataFrame(d['general_train_df'].columns.delete(0))
##coeff_df.columns = ['Feature']
##coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
##
##print("coeff_df.sort_values(by='Correlation', ascending=False"+str(coeff_df.sort_values(by='Correlation', ascending=False)))
##
#### Support Vector Machines
###
###svc = SVC()
###svc.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = svc.predict(X_gen_test)
###acc_svc = round(svc.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_svc'+str(acc_svc))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_svc1'+str(actual_accuracy))
###
###
###knn = KNeighborsClassifier(n_neighbors = 3)
###knn.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = knn.predict(X_gen_test)
###acc_knn = round(knn.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_knn'+str(acc_knn))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_knn'+str(actual_accuracy))
###
#### Gaussian Naive Bayes
###
###gaussian = GaussianNB()
###gaussian.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = gaussian.predict(X_gen_test)
###acc_gaussian = round(gaussian.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_gaussian'+str(acc_gaussian))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_gauss'+str(actual_accuracy))
###
#### Perceptron
###
###perceptron = Perceptron()
###perceptron.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = perceptron.predict(X_gen_test)
###acc_perceptron = round(perceptron.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_perceptron'+str(acc_perceptron))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_perceptron'+str(actual_accuracy))
###
#### Linear SVC
###
###linear_svc = LinearSVC()
###linear_svc.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = linear_svc.predict(X_gen_test)
###acc_linear_svc = round(linear_svc.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_linear_svc'+str(acc_linear_svc))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_svc'+str(actual_accuracy))
###
#### Stochastic Gradient Descent
###
###sgd = SGDClassifier()
###sgd.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = sgd.predict(X_gen_test)
###acc_sgd = round(sgd.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_sgd'+str(acc_sgd))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_sgd'+str(actual_accuracy))
###
#### AdaBoost
###
###Ada_Boost = AdaBoostClassifier(n_estimators=100)
###Ada_Boost.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = Ada_Boost.predict(X_gen_test)
###Ada_Boost.score(X_gen_train, Y_gen_train)
###acc_Ada_Boost = round(Ada_Boost.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_Ada_Boost'+str(acc_Ada_Boost))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_Ada_Boost'+str(actual_accuracy))
###
#### Decision Tree
###
###decision_tree = DecisionTreeClassifier()
###decision_tree.fit(X_gen_train, Y_gen_train)
###Y_gen_pred = decision_tree.predict(X_gen_test)
###acc_decision_tree = round(decision_tree.score(X_gen_train, Y_gen_train) * 100, 2)
###print('acc_decision_tree'+str(acc_decision_tree))
###actual_accuracy = accurate_to(Y_gen_pred, 0)
###print('actual_accuracy_tree'+str(actual_accuracy))
##
##
### Random Forest
##
##random_forest = RandomForestClassifier(n_estimators=100)
##random_forest.fit(X_gen_train, Y_gen_train)
##Y_gen_predz = random_forest.predict(X_gen_test)
##random_forest.score(X_gen_train, Y_gen_train)
##acc_random_forest = round(random_forest.score(X_gen_train, Y_gen_train) * 100, 2)
##print('acc_random_forest'+str(acc_random_forest))
##actual_accuracy = accurate_to(Y_gen_predz, 0)
##print('actual_accuracy_forrest'+str(actual_accuracy))
##
###models = pd.DataFrame({
###    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
###              'Random Forest', 'Naive Bayes', 'Perceptron',
###              'Stochastic Gradient Decent', 'Linear SVC',
###              'Decision Tree', 'AdaBoost'],
###    'Score': [acc_svc, acc_knn, acc_log,
###              acc_random_forest, acc_gaussian, acc_perceptron,
###              acc_sgd, acc_linear_svc, acc_decision_tree, acc_Ada_Boost]})
###print("models.sort_values(by='Score', ascending=False)"+str(models.sort_values(by='Score', ascending=False)))
##
###submission3 = pd.DataFrame({
###        "PassengerId": test_df["PassengerId"],
###        "Survived": Y_gen_pred
###    })
###submission3
