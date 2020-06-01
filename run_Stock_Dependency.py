# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:15:36 2020

@author: timot
"""


from Stock_Dependency_Analyzer import Stock_Dependency_Analyzer

#%% Inputs
doHTML = False

# Stock data
analyzeTicker = 'TSLA'
metricTickers = ['GOLD','XOM']
years = 1

# Correlator data
trainSize = 0.8
analyzeInterval = 3
metricInterval = 7
changeFilter = 0.02

# SVM parameters:
SVM_kernel = 'sigmoid' # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
SVM_degree = 3
SVM_gamma = 'scale' # {'scale', 'auto'}

# KNN parameters:
KNN_neighbors = 4
KNN_weights = 'uniform' # {‘uniform’, ‘distance’}

# RF parameters:
RF_n_estimators = 100
RF_criterion = 'entropy' # {'gini', 'entropy'}

#%% Execute

dependency_analyzer = Stock_Dependency_Analyzer()
dependency_analyzer.collect_data(analyzeTicker, metricTickers, years)
dependency_analyzer.build_correlation(analyzeInterval, metricInterval, changeFilter=changeFilter, doHTML=doHTML)
report, p = dependency_analyzer.create_all_classifiers(SVM_kernel, SVM_degree, SVM_gamma, \
                                           KNN_neighbors, KNN_weights, \
                                               RF_n_estimators, RF_criterion, \
                                               trainSize=trainSize, doHTML=doHTML)
dependency_analyzer.run_prediction()