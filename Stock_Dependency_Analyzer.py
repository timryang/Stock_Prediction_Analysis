# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:32:15 2020

@author: timot
"""

from CommonFunctions.commonFunctions import *
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#%% Stock dependency class

class Stock_Dependency_Analyzer():
    
    def __init__(self):
        self.analyzeTicker_ = ''
        self.metricTickers_ = []
        self.analyzerStockData_ = pd.DataFrame()
        self.metricStockData_ = pd.DataFrame()
        self.stockResults_ = np.array([])
        self.metric_filt_ = np.array([])
        self.predictors_ = np.array([])
        self.lr_clf_ = LogisticRegression()
        self.svm_clf_ = svm.SVC()
        self.knn_clf_ = KNeighborsClassifier()
        self.rf_clf_ = RandomForestClassifier()
        
    def collect_data(self, analyzeTicker, metricTickers, years):
        self.analyzerStockData_ = collect_stock_data(analyzeTicker, years)
        self.metricStockData_ = [collect_stock_data(ticker, years) for ticker in metricTickers]
        self.analyzeTicker_ = analyzeTicker
        self.metricTickers_ = metricTickers
        
    def load_data(self, analyzerDir, metricDir, analyzeTicker, metricTickers):
        self.analyzeTicker_ = analyzeTicker
        self.metricTickers_ = metricTickers
        if isinstance(analyzerDir, pd.DataFrame):
            self.analyzerStockData_ = analyzerDir
            self.metricStockData_ = metricDir
        else:
            self.analyzerStockData_ = pd.read_csv(analyzerDir)
            self.analyzerStockData_ = pd.read_csv(metricDir)
    
    def build_correlation(self, analyzeInterval, metricInterval, changeFilter=0, doHTML=False):
        
        # Vectorize data
        analyzeCloseData = self.analyzerStockData_['Close'].values
        metricCloseData = self.metricStockData_[0]['Close'].values
        for idx in range(1, len(self.metricStockData_)):
            metricCloseData = np.column_stack((metricCloseData, self.metricStockData_[idx]['Close'].values))
        
        # Get dates
        dates = pd.to_datetime(self.analyzerStockData_['Date'])
        
        # Compute metric percent change
        metricDates = dates[metricInterval:]
        metricCloseDiff = np.array([metricCloseData[i+metricInterval]-metricCloseData[i] \
                                    for i in range(metricCloseData.shape[0]-metricInterval)])
        metricPercDiff = metricCloseDiff / metricCloseData[:-metricInterval]
        
        # Compute analyze percent change
        analyzeDates = dates[metricInterval:-analyzeInterval]
        analyzeCloseDiff = np.array([analyzeCloseData[i+metricInterval+analyzeInterval]-val \
                                         for i, val in enumerate(analyzeCloseData[metricInterval:-analyzeInterval])])
        analyzePercDiff = analyzeCloseDiff / analyzeCloseData[metricInterval:-analyzeInterval]
        
        # Filter on changeFilter
        validIdx = np.where(np.abs(analyzePercDiff) > changeFilter)[0]
        analyze_filt = analyzePercDiff[validIdx]
        metric_filt = metricPercDiff[validIdx]
        if metric_filt.ndim == 1:
            self.metric_filt_ = metric_filt.reshape(-1, 1)
        else:
            self.metric_filt_ = metric_filt
        
        # Assign labels
        stockResults = np.where(analyze_filt > 0, 'Positive', 'Negative')
        self.stockResults_ = stockResults
        
        # Assign predictors
        predictors = metricPercDiff[len(analyzePercDiff):]
        if predictors.ndim == 1:
            self.predictors_ = predictors.reshape(-1, 1)
        else:
            self.predictors_ = predictors
        
        # Plot close vs dates
        x_values = [dates]
        y_values = [analyzeCloseData]
        labels = [self.analyzeTicker_]
        if len(self.metricTickers_) < 2:
            y_values.append(metricCloseData)
            labels.append(self.metricTickers_[0])
            x_values.append(dates)
        else:
            for idx, val in enumerate(self.metricTickers_):
                y_values.append(metricCloseData[:,idx])
                labels.append(val)
                x_values.append(dates)
        title = 'Close Data'
        x_label = 'Dates'
        y_label = 'Close Price'
        p = plot_values(x_values, y_values, labels, x_label, y_label, title, isDates=True, isBokeh=doHTML)
        
        # Plot diff vs dates
        x_values_diff = [analyzeDates]
        y_values_diff = [analyzePercDiff]
        if len(self.metricTickers_) < 2:
            y_values_diff.append(metricPercDiff)
            x_values_diff.append(metricDates)
        else:
            for idx, val in enumerate(self.metricTickers_):
                x_values_diff.append(metricDates)
                y_values_diff.append(metricPercDiff[:,idx])
        title_diff = 'Interval % Change'
        y_label_diff = '% Change'
        p_delta = plot_values(x_values_diff, y_values_diff, labels, x_label, y_label_diff, title_diff, isDates=True, isBokeh=doHTML)
        
        if doHTML:
            p = bokeh.layouts.row(p, p_delta)

        # Plot scatter
        if len(self.metricTickers_) <= 3:
            titleTxt = self.analyzeTicker_ + " % Change"
            axes_labels = [val + " % Change" for val in self.metricTickers_]
            if len(self.metricTickers_) == 1:
                axes_labels.append(self.analyzeTicker_ + " % Change")
            ps = plot_multi_scatter([metric_filt, predictors], ['Historical', 'Predictors'],\
                               axes_labels, self.analyzeTicker_ + " Result", \
                                   color_values_list=[analyze_filt, False], saveFig=doHTML)
        else:
            ps = "Too many to plot"
        
        return p, ps
        
    def create_all_classifiers(self, SVM_kernel, SVM_degree, SVM_gamma,\
                               KNN_neighbors, KNN_weights,\
                                   RF_n_estimators, RF_criterion,
                                   trainSize=0.8, doHTML=False):
        
        # Train test split
        xTrain, xTest, yTrain, yTest = train_test_split(self.metric_filt_, self.stockResults_,\
                                                train_size=trainSize, random_state=42)
            
        # Logistic regression
        self.lr_clf_ = LogisticRegression().fit(xTrain, yTrain)
        print("Logistic Regression Results:")
        report_lr, p_lr = classifier_statistics(xTest, yTest, self.lr_clf_, doHTML=doHTML)
        print("\n")
        
        # SVM
        self.svm_clf_ = svm.SVC(kernel=SVM_kernel, degree=SVM_degree, gamma=SVM_gamma).fit(xTrain, yTrain)
        print("SVM Results:")
        report_svm, p_svm = classifier_statistics(xTest, yTest, self.svm_clf_, doHTML=doHTML)
        print("\n")
        
        # KNN
        self.knn_clf_ = KNeighborsClassifier(n_neighbors=KNN_neighbors, weights=KNN_weights)
        self.knn_clf_.fit(xTrain, yTrain)
        print("KNN Results:")
        report_knn, p_knn = classifier_statistics(xTest, yTest, self.knn_clf_, doHTML=doHTML)
        print("\n")
        
        # RF
        self.rf_clf_ = RandomForestClassifier(n_estimators=RF_n_estimators, criterion=RF_criterion)
        self.rf_clf_.fit(xTrain, yTrain)
        print("Random Forest Results:")
        report_rf, p_rf = classifier_statistics(xTest, yTest, self.rf_clf_, doHTML=doHTML)
        print("\n")
        
        report_list = [report_lr, report_svm, report_knn, report_rf]
        p_list = [p_lr, p_svm, p_knn, p_rf]
        
        return report_list, p_list
        
    def run_prediction(self):
        lr_predict = self.lr_clf_.predict(self.predictors_)
        svm_predict = self.svm_clf_.predict(self.predictors_)
        knn_predict = self.knn_clf_.predict(self.predictors_)
        rf_predict = self.rf_clf_.predict(self.predictors_)
        
        reportDF = pd.DataFrame()
        reportDF['Days From Today'] = pd.Series(list(range(1,len(lr_predict)+1)))
        reportDF['Log Reg'] = pd.Series(lr_predict)
        reportDF['SVM'] = pd.Series(svm_predict)
        reportDF['KNN'] = pd.Series(knn_predict)
        reportDF['Rand Forest'] = pd.Series(rf_predict)
        
        print(reportDF)
        
        return reportDF