# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:15:36 2020

@author: timot
"""


from Stock_Dependency_Analyzer.Stock_Dependency_Analyzer import Stock_Dependency_Analyzer

#%% Inputs
doHTML = False

# Stock data
analyzeTicker = 'TSLA'
metricTickers = ['GOLD','XOM','MRNA','GOOG']
years = 1

# Correlator data
trainSize = 0.8
analyzeInterval = 3
metricInterval = 7
changeFilter = 0

# SVM parameters:
SVM_kernel = 'sigmoid' # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
SVM_degree = 3
SVM_gamma = 'scale' # {'scale', 'auto'}

# KNN parameters:
KNN_neighbors = 4
KNN_weights = 'uniform' # {‘uniform’, ‘distance’}

#%% Execute

dependency_analyzer = Stock_Dependency_Analyzer()
dependency_analyzer.collect_data(analyzeTicker, metricTickers, years)
dependency_analyzer.build_correlation(analyzeInterval, metricInterval, changeFilter=changeFilter, doHTML=doHTML)
report, p = dependency_analyzer.create_all_classifiers(SVM_kernel, SVM_degree, SVM_gamma, \
                                           KNN_neighbors, KNN_weights, \
                                               trainSize=trainSize, doHTML=doHTML)
dependency_analyzer.run_prediction()