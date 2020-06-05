# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:15:36 2020

@author: timot
"""


from Stock_Dependency_Analyzer import Stock_Dependency_Grid_Search

#%% Inputs
doHTML = True

# Stock data
analyzeTicker = 'DAL'
metricTickers = ['NDAQ','XOM']
years = 3

# Correlator data
trainSize = [0.8]
kFold = [5]
analyzeInterval = [1,2,3]
metricInterval = [4,5,6,7]
changeFilter = [0,0.01,0.02]

# SVM parameters:
scaleSVM = True
SVM_c = [1]
SVM_kernel = ['rbf'] # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}
SVM_degree = [3]
SVM_gamma = ['scale'] # {'scale', 'auto', float}
SVM_coeff = [0] # used with poly / sigmoid

# KNN parameters:
KNN_neighbors = [1, 4, 9]
KNN_weights = ['distance', 'uniform'] # {‘uniform’, ‘distance’}

# RF parameters:
RF_n_estimators = [100]
RF_criterion = ['entropy'] # {'gini', 'entropy'}

#%% Execute

SVM_grid = {'C': SVM_c, 'kernel': SVM_kernel, 'degree': SVM_degree, 'gamma': SVM_gamma, 'coef0': SVM_coeff}
KNN_grid = {'n_neighbors': KNN_neighbors, 'weights': KNN_weights}
RF_grid = {'n_estimators': RF_n_estimators, 'criterion': RF_criterion}

best_p, best_ps, best_scores, best_report, best_conf_mat, best_svm_params, best_knn_params, best_rf_params, best_result,\
        best_aInt, best_mInt, best_change, best_train, best_k = Stock_Dependency_Grid_Search(analyzeTicker, metricTickers, years, trainSize, kFold, analyzeInterval, metricInterval, changeFilter,\
                                 scaleSVM, SVM_grid, KNN_grid, RF_grid, doHTML)