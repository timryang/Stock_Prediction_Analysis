# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:50:28 2020

@author: timot
"""
from Stock_NB_Analyzer import *
from nltk.corpus import stopwords

#%% Inputs:
    
doHTML = True

ticker = 'MRNA'
years = 1

# Get new if false
loadTweets = True
loadData = True

# Classifier tweet parameters (only used if loadTweets = False):
userName = None
geoLocation = None
distance = None
sinceDate = '2020-03-01'
untilDate = '2020-05-14'
querySearch = 'MRNA'
maxTweets = 500
topTweets = True
lang = 'en'

# Tweet and data directories:
tweetDir = './CSV_Files/MRNA_Tweets.csv'
dataDir = './CSV_Files/MRNA.csv'

# Correlation parameters
deltaInterval = [1,2] # days
changeFilter = [0,0.02] # %

# Classifier parameters
trainSize = [0.8]
stopwordsList = [None, stopwords.words('english')]
useIDF = [True]
do_downsample = [True]
do_stat = True
numFeatures = 10

# New tweet prediction parameters:
userName_predict = None
geoLocation_predict = None
distance_predict = None
txtSearch_predict = 'MRNA'
numMaxTweets_predict = 10
topTweets_predict = True
lang_predict = 'en'
printAll = True


#%% Execute

NB_analyzer = Stock_NB_Analyzer()
if loadTweets:
    NB_analyzer.load_tweets(tweetDir)
else:
    NB_analyzer.collect_tweets(userName, geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets, lang)
if loadData:
    NB_analyzer.load_data(dataDir)
else:
    NB_analyzer.collect_data(ticker, years)

NB_analyzer, count_report, p, report, most_inform, conf_mat, deltaInterval, changeThreshold, trainSize, useIDF, do_downsample, stopwordsList = \
        Stock_NB_Grid_Search(NB_analyzer, trainSize, deltaInterval, changeFilter, useIDF, do_downsample, stopwordsList, do_stat, numFeatures, doHTML)

NB_analyzer.run_prediction(userName_predict, geoLocation_predict, distance_predict, txtSearch_predict,\
                           numMaxTweets_predict, topTweets_predict, lang_predict, printAll)