# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:32:15 2020

@author: timot
"""

from CommonFunctions.commonFunctions import *
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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
        self.scaleTF_ = StandardScaler()
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
        self.scaleTF_ = StandardScaler().fit(metricPercDiff)
        
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
            p = bokeh.layouts.row(p, p_delta, sizing_mode='stretch_both')

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
        
    def create_all_classifiers(self, scaleSVM=True, SVM_grid=None, KNN_grid=None, RF_grid=None, \
                                   trainSize=0.8, k=5, doHTML=False):
        
        # Train test split
        xTrain, xTest, yTrain, yTest = train_test_split(self.metric_filt_, self.stockResults_,\
                                                train_size=trainSize, random_state=0)    
        
        # SVM
        if scaleSVM:
            xTrain_svm = self.scaleTF_.transform(xTrain)
            xTest_svm = self.scaleTF_.transform(xTest)
        else:
            xTrain_svm = xTrain
            xTest_svm = xTest
        svm_clf = svm.SVC()
        svm_clf = GridSearchCV(svm_clf, SVM_grid, cv=k)
        svm_clf.fit(xTrain_svm, yTrain)
        self.svm_clf_, svm_params, scores_svm = extract_from_gridCV(svm_clf, k)
        if not doHTML:
            print("SVM Results:")
        report_svm, p_svm = classifier_statistics(xTest_svm, yTest, self.svm_clf_, doHTML=doHTML)
        
        # KNN
        knn_clf = KNeighborsClassifier()
        knn_clf = GridSearchCV(knn_clf, KNN_grid, cv=k)
        knn_clf.fit(xTrain, yTrain)
        self.knn_clf_, knn_params, scores_knn = extract_from_gridCV(knn_clf, k)
        if not doHTML:
            print("KNN Results:")
        report_knn, p_knn = classifier_statistics(xTest, yTest, self.knn_clf_, doHTML=doHTML)
        
        # RF
        rf_clf = RandomForestClassifier()
        rf_clf = GridSearchCV(rf_clf, RF_grid, cv=k)
        rf_clf.fit(xTrain, yTrain)
        self.rf_clf_, rf_params, scores_rf = extract_from_gridCV(rf_clf, k)
        if not doHTML:
            print("Random Forest Results:")
        report_rf, p_rf = classifier_statistics(xTest, yTest, self.rf_clf_, doHTML=doHTML)
        
        cv_scores = [scores_svm, scores_knn, scores_rf]
        report_list = [report_svm, report_knn, report_rf]
        p_list = [p_svm, p_knn, p_rf]
        
        return cv_scores, report_list, p_list, svm_params, knn_params, rf_params
        
    def run_prediction(self, scaleSVM):
        if scaleSVM:
            predictors_svm = self.scaleTF_.transform(self.predictors_)
        else:
            predictors_svm = self.predictors_
        svm_predict = self.svm_clf_.predict(predictors_svm)
        knn_predict = self.knn_clf_.predict(self.predictors_)
        rf_predict = self.rf_clf_.predict(self.predictors_)
        
        reportDF = pd.DataFrame()
        reportDF['Days From Today'] = pd.Series(list(range(1,len(svm_predict)+1)))
        reportDF['SVM'] = pd.Series(svm_predict)
        reportDF['KNN'] = pd.Series(knn_predict)
        reportDF['Rand Forest'] = pd.Series(rf_predict)
        
        print(reportDF)
        
        return reportDF
    
#%% Grid Search Function

def Stock_Dependency_Grid_Search(dependency_analyzer, trainSize, kFold, analyzeInterval, metricInterval, changeFilter,\
                                 scaleSVM, SVM_grid, KNN_grid, RF_grid, doHTML):
    
    best_score = 0
    
    for i_aInt in analyzeInterval:
        for i_mInt in metricInterval:
            for i_change in changeFilter:
                p, ps = dependency_analyzer.build_correlation(i_aInt, i_mInt, changeFilter=i_change, doHTML=doHTML)
                for i_train in trainSize:
                    for i_k in kFold:
                        scores, report, conf_mat, svm_params, knn_params, rf_params = \
                            dependency_analyzer.create_all_classifiers(scaleSVM=scaleSVM, SVM_grid=SVM_grid, KNN_grid=KNN_grid, RF_grid=RF_grid,\
                                                                       trainSize=i_train, k=i_k, doHTML=doHTML)
                        results = dependency_analyzer.run_prediction(scaleSVM)
                        
                        report_scores = [temp['F1'].iloc[np.where(temp['Class/Metric'].values=='accuracy')[0][0]] for temp in report]
                        max_score = max(report_scores)
                        
                        if (max_score > best_score):
                            best_score = max_score
                            best_p = p
                            best_ps = ps
                            best_scores = scores
                            best_report = report
                            best_conf_mat = conf_mat
                            best_svm_params = svm_params
                            best_knn_params = knn_params
                            best_rf_params = rf_params
                            best_result = results
                            best_aInt = i_aInt
                            best_mInt = i_mInt
                            best_change = i_change
                            best_train = i_train
                            best_k = i_k
                            
    return best_p, best_ps, best_scores, best_report, best_conf_mat, best_svm_params, best_knn_params, best_rf_params, best_result,\
        best_aInt, best_mInt, best_change, best_train, best_k