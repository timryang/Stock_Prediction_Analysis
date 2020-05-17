# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:08:39 2020

@author: Tim
"""
# This script is more user friendly which uses the classes/functions in
# stock_tweet_NB

from stock_tweet_NB import stock_NB_Tweet_Analyzer
import pandas as pd
from nltk.corpus import stopwords

#%% User Inputs:

# Give ticker symbol for company under analysis
ticker = 'MRNA'

# Parameters to gather tweets to build model
# Currently supports one year history
classifyTxtSearch = 'MRNA' # Search query
classifyNumMaxTweets = 0 # Set to zero to gather all tweets
classifyStartDate = '2020-03-01'
classifyStopDate = '2020-05-14'
classifyTopTweets = True # Gather top tweets if true (all tweets if false)

# Parameter to adjust model creation
trainSize = 0.8 # Split percentage between train and test sets
stopwordsList = stopwords.words('english') # Words to filter out from model
useIDF = True # Apply less weighting to highly frequent words across tweets

# Parameter to analyze model data
plotRetweets = True # Plot distribution of retweets if true
plotCloseData = True # Plot stock close price over time if true
plotDeltaClose = True # Plot change in stock price
numFeatures = 10 # Number of words to display to shows it's "informativeness"

# Parameters to gather tweets to predict stock performance
predictTxtSearch = 'MRNA' # Search query
predictNumMaxTweets = 10 # Qty of recent tweets
predictTopTweets = True # Gather top tweets if true (all tweets if false)
printAll = False # Print each tweet and resulting prediction if true

#%% Everyting below executes using the inputs above
# Create Analyzer
stockAnalyzer = stock_NB_Tweet_Analyzer(ticker)

# Collect Historical Tweets
stockAnalyzer.collect_tweets(classifyTxtSearch, startDate = classifyStartDate,\
                             stopDate = classifyStopDate,\
                                    numMaxTweets = classifyNumMaxTweets,\
                                        topTweets = classifyTopTweets)

# Correlate Historical Tweets to Stock Performance
stockAnalyzer.correlate_tweets()

# Create Classifier to Predict Tweets
stockAnalyzer.create_classifier(trainSize, stopwordsList, useIDF)

# Analyze Classifier Data
stockAnalyzer.plot_tweet_data(retweets = plotRetweets, closeData = plotCloseData, deltaClose = plotDeltaClose)
stockAnalyzer.show_most_informative(n = numFeatures)

# Create Prediction from New Tweets
stockAnalyzer.predict(predictTxtSearch, predictNumMaxTweets, predictTopTweets, printAll)