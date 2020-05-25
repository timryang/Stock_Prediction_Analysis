# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:50:28 2020

@author: timot
"""
from Stock_NB_Analyzer import Stock_NB_Analyzer
from nltk.corpus import stopwords

#%% Inputs:

ticker = 'MRNA'
years = 3

# Get new if false
loadTweets = True
loadData = True

# Classifier tweet parameters (only used if loadTweets = False):
geoLocation = None
distance = None
sinceDate = '2020-03-01'
untilDate = '2020-05-14'
querySearch = 'MRNA'
maxTweets = 0
topTweets = True

# Tweet and data directories:
tweetDir = './CSV_Files/MRNA_Tweets.csv'
dataDir = './CSV_Files/MRNA.csv'

# Change interval
deltaInterval = 2 # days

# Classifier parameters
trainSize = 0.8
stopwordsList = stopwords.words('english')
useIDF = True
do_downsample = True
do_stat = True
numFeatures = 10

# New tweet prediction parameters:
geoLocation_predict = None
distance_predict = None
txtSearch_predict = 'MRNA'
numMaxTweets_predict = 10
topTweets_predict = True
printAll = True


#%% Execute

NB_analyzer = Stock_NB_Analyzer()
if loadTweets:
    NB_analyzer.load_tweets(tweetDir)
else:
    NB_analyzer.collect_tweets(geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets)
if loadData:
    NB_analyzer.load_data(dataDir)
else:
    NB_analyzer.collect_data(ticker, years)
NB_analyzer.correlate_tweets(deltaInterval)
NB_analyzer.plot_data(deltaInterval)
NB_analyzer.create_classifier(trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures)
NB_analyzer.run_prediction(geoLocation_predict, distance_predict, txtSearch_predict,\
                           numMaxTweets_predict, topTweets_predict, printAll)