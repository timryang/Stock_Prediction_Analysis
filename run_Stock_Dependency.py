# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:15:36 2020

@author: timot
"""


from Stock_Dependency_Analyzer import Stock_Dependency_Analyzer

#%% Inputs
doHTML = False

# Stock data
analyzeTicker = 'DAL'
metricTickers = ['NDAQ','XOM']
years = 3

# Correlator data
trainSize = 0.8
kFold = 5
analyzeInterval = 3
metricInterval = 7
changeFilter = 0.02

# SVM parameters:
scaleSVM = True
SVM_c = 1
SVM_kernel = 'rbf' # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
SVM_degree = 3
SVM_coeff = 0 # using with poly / sigmoid
SVM_gamma = 'scale' # {'scale', 'auto', float}

# KNN parameters:
KNN_neighbors = 10
KNN_weights = 'distance' # {‘uniform’, ‘distance’}

# RF parameters:
RF_n_estimators = 100
RF_criterion = 'entropy' # {'gini', 'entropy'}

#%% Execute

dependency_analyzer = Stock_Dependency_Analyzer()
dependency_analyzer.collect_data(analyzeTicker, metricTickers, years)
dependency_analyzer.build_correlation(analyzeInterval, metricInterval, changeFilter=changeFilter, doHTML=doHTML)
scores, report, p = dependency_analyzer.create_all_classifiers(scaleSVM=scaleSVM, c_svm=SVM_c, SVM_kernel=SVM_kernel,\
                                                       SVM_degree=SVM_degree, coeff_svm=SVM_coeff, SVM_gamma=SVM_gamma, \
                                                           KNN_neighbors=KNN_neighbors, KNN_weights=KNN_weights, \
                                                               RF_n_estimators=RF_n_estimators, RF_criterion=RF_criterion, \
                                                               trainSize=trainSize, k=kFold, doHTML=doHTML)
dependency_analyzer.run_prediction(scaleSVM)