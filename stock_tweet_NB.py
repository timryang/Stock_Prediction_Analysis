# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:38:49 2020

@author: Tim
"""
# This script creates a class, stock_NB_Tweet_Analyzer, to collect historical
# tweets used to generate a Naive Bayes machine learning model to predict next
# day's stock performance. An example of using the class is shown in the "main"

# Script developed using Python 3.7

#%% Import all necessary libraries
import GetOldTweets3 as got
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter

#%% Functions

def find_idx(inputList, condition):
    return [idx for idx, val in enumerate(inputList) if val == condition]

def date_check(stockData):
    # Confirms that all dates within stock data csv file are unique
    dateList = list(stockData.Date)
    if len(list(set(dateList))) != len(dateList):
        raise ValueError("There are redundant dates in stock data")
    else:
        print('All dates in stock data are unique')
        
def transform_text(text, count_vect, tfTransformer):
    # count_vect converts text into array format represeting word count
    # tfTransformer normalizes word counts to account for term frequency within document
    count_text = count_vect.transform(text)
    tf_text = tfTransformer.transform(count_text)
    return tf_text

def get_tweets(txtSearch, startDate = None, stopDate = None, geoLocation = None,\
               distance = None, topTweets = True, numMaxTweets = 10):
    # Using open source library (got) to collect tweets based on criteria
    # Returns tweets in DataFrame format consisting of Date, Text, # of Retweets
    if (startDate == None and geoLocation == None):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    elif (geoLocation == None and startDate != None):
        tweetCriteria = got.manager.TweetCriteria().setSince(startDate)\
                                                .setUntil(stopDate)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    elif (startDate == None and geoLocation != None):
        tweetCriteria = got.manager.TweetCriteria().setNear(geoLocation)\
                                                .setWithin(distance)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    else:
        tweetCriteria = got.manager.TweetCriteria().setSince(startDate)\
                                                .setUntil(stopDate)\
                                                .setNear(geoLocation)\
                                                .setWithin(distance)\
                                                .setQuerySearch(txtSearch)\
                                                .setTopTweets(topTweets)\
                                                .setMaxTweets(numMaxTweets)
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    tweetsParsed = [[tweet.date, tweet.text, tweet.retweets] for tweet in tweets]
    tweetsDF = pd.DataFrame(tweetsParsed, columns = ['Date', 'Text', 'Retweets'])
    tweetsDF.sort_values(by = ['Date'], inplace = True)
    tweetsDF.reset_index(drop = True, inplace = True)
    return tweetsDF

#%% Create Class

class stock_NB_Tweet_Analyzer:
    
    def __init__(self, stockData):
        # Load historical stock data in csv format
        # Expected format is from Yahoo's historical stock price data
        # An example csv can be downloaded here:
        # https://finance.yahoo.com/quote/TSLA/history?p=TSLA
        
        # Confirm all dates are unique in historical stock data
        date_check(stockData)
        # Initialize variables
        self.stockData_ = stockData
        self.totalTweets_ = []
        self.totalResults_ = []
        self.classifierTweets_ = pd.DataFrame()
        self.countVect_ = CountVectorizer()
        self.tfTransformer_ = TfidfTransformer()
        self.clf_ = MultinomialNB()
        self.ratio_dict_ = {}
        self.predictTweets_ = pd.DataFrame()
        self.predictions_ = np.array([])
        
    def collect_tweets(self, txtSearch, startDate = None, stopDate = None,\
                       geoLocation = None, distance = None, topTweets = True,\
                           numMaxTweets = 10):
        self.classifierTweets_ = get_tweets(txtSearch, startDate, stopDate,\
                                                geoLocation, distance,\
                                                    topTweets, numMaxTweets)
        self.totalTweets_ = list(self.classifierTweets_['Text']) # Store text in list

    def plot_tweet_data(self, retweets = True, closeData = True, deltaClose = True):
        dates = pd.to_datetime(self.stockData_['Date'])
        if retweets:
            # Show distribution of tweets by # of retweets
            plt.hist(list(self.classifierTweets_['Retweets']), bins = range(self.classifierTweets_['Retweets'].max()))
            plt.title('Distribution of Retweets')
            plt.xlabel('# of Retweets')
            plt.ylabel('Occurences')
            plt.show()
        if closeData:
            # Plot historical close data over entire dataset
            ax = plt.gca()
            formatter = mdates.DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(formatter)
            locator = mdates.DayLocator(bymonthday = [1, 15])
            ax.xaxis.set_major_locator(locator)
            plt.plot_date(dates, self.stockData_['Close'].values, '-')
            plt.title('Daily Close Performance')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.xticks(rotation = 70)
            plt.show()
        if deltaClose:
            # Plot historical delta close data
            deltaClose = np.diff(self.stockData_['Close'].values)
            ax = plt.gca()
            formatter = mdates.DateFormatter("%m-%d")
            ax.xaxis.set_major_formatter(formatter)
            locator = mdates.DayLocator(bymonthday = [1, 15])
            ax.xaxis.set_major_locator(locator)
            plt.plot_date(dates[:-1], deltaClose, '-')
            plt.title('Daily Change Performance')
            plt.xlabel('Date')
            plt.ylabel('Change')
            plt.xticks(rotation = 70)
            plt.show()
            
    def correlate_tweets(self):
        # Correlate each tweet to the next day's stock performance
        
        tweetDatesAll = [dt.date() for dt in self.classifierTweets_['Date']]
        stockDates = list(self.stockData_['Date'])
        deltaClose = np.diff(self.stockData_['Close'].values)
        resultsClose = np.where(deltaClose > 0, 'positive', 'negative')
        for idx, tweetDate in enumerate(tweetDatesAll):
            # Check if the day of tweet and the next day is a weekday
            # (No stock data on weekends)
            if tweetDate.weekday() >= 4:
                # If not, tweet day is correlated to the nearest previous workday
                # And next day is correlated to the nearest next workday
                currentStockDay = tweetDate - datetime.timedelta(days = (tweetDate.weekday() - 4))
            else:
                currentStockDay = tweetDate
            currentStockDayStr = str(currentStockDay)
            # While loop accounts for tweets that fall on a weekday holiday
            # (No stock data on holidays)
            while currentStockDayStr not in stockDates:
                print("Current date, " + currentStockDayStr + " was not found. Decrementing...")
                currentStockDay = currentStockDay - datetime.timedelta(days = 1)
                currentStockDayStr = str(currentStockDay)
            # Get price change
            result = resultsClose[find_idx(stockDates, currentStockDayStr)]
            # Store corresponding stock performance in list
            self.totalResults_.append(result[0])
        print("Total positive tweets: " + str(self.totalResults_.count('positive')))
        print("Total negative tweets: " + str(self.totalResults_.count('negative')))
    
    def create_classifier(self, trainSize, stopwordsList, useIDF):
        # Creates the ML algorithm based on tweets / corresponding stock data
        
        # Splits the historical tweet data into train and test sets
        # Takes user input (trainSize) to define qty split between train/test
        tweetTxtTrain, tweetTxtTest, resultsTrain, resultsTest = train_test_split(
            self.totalTweets_, self.totalResults_, train_size = trainSize, random_state = 42)
        # Transforms tweets into an array corresponding to word counts
        # Takes user input (stopwordsList) to ignore words in the list - set to None to use all
        self.count_vect_ = CountVectorizer(stop_words = stopwordsList)
        train_counts = self.count_vect_.fit_transform(tweetTxtTrain)
        # Normalizes word counts by word frequency within each tweet
        # Takes user input (useIDF) to give less weight to
        # highly frequent terms across all tweets if true
        self.tfTransformer_ = TfidfTransformer(use_idf = useIDF)
        train_tf = self.tfTransformer_.fit_transform(train_counts)
        # Create multinomial Naive Bayes model (binomial in this case) using historical data
        # argmax_y(P(y|xi..xn) = P(xi|y)*...*P(xn|y(*P(y))
        # y is [negative, positive], xi is each word within all tweets
        self.clf_ = MultinomialNB().fit(train_tf, resultsTrain)
        # Test model and print accuracy given historical data
        test_tf = transform_text(tweetTxtTest, self.count_vect_, self.tfTransformer_)
        print("Accuracy: %0.2f" % (self.clf_.score(test_tf, resultsTest)*100))

    def show_most_informative(self, n = 10):
        # Shows statistics on the historical data used to generate ML model
        
        classes = self.clf_.classes_
        features = self.count_vect_.get_feature_names()
        # Get P(word|outcome) for every word and outcome
        probabilities = np.exp(self.clf_.feature_log_prob_)
        # Get P(word|outcome_x)/P(word|outcome_y) for every word
        one2two_ratio = np.round(np.divide(probabilities[0], probabilities[1]), 3)
        # Get P(word|outcome_y)/P(word|outcome_x) for every word
        two2one_ratio = np.round(np.divide(probabilities[1], probabilities[0]), 3)
        # Get highest ratios - takes user input (n) for qty to print
        top_one2two = (sorted(zip(features, one2two_ratio), key = itemgetter(1)))[:-(n+1):-1]
        top_two2one = (sorted(zip(features, two2one_ratio), key = itemgetter(1)))[:-(n+1):-1]
        # Get outcome labels
        label_one2two = classes[0] + ':' + classes[1] 
        label_two2one = classes[1] + ':' + classes[0]
        # Print dictionary with ratio results - this indicates the most "informative" words
        self.ratio_dict_ = {label_one2two: top_one2two, label_two2one: top_two2one}
        print("\nBelow printout gives the most informative words.")
        print("Example -> neg:pos: ('gain', 3.0) indicates 'gain'"\
              + "is 3.0x more likely to appear in a neg tweet vs pos tweet.\n")
        print("{:<25s} {:<25s}".format(label_one2two, label_two2one))
        for one, two in zip(top_one2two, top_two2one):
            print("{:<25s} {:<25s}".format(str(one), str(two)))

    def predict(self, txtSearch, numTestMaxTweets, topTestTweets, printAll):
        # Get a new set of recent tweets and predict next stock day's performance
        
        self.predictTweets_ = get_tweets(txtSearch, numMaxTweets = numTestMaxTweets, topTweets = topTestTweets)
        tf_text = transform_text(self.predictTweets_.Text, self.count_vect_, self.tfTransformer_)
        self.predictions_ = self.clf_.predict(tf_text)
        if printAll:
            for idx, prediction in enumerate(self.predictions_):
                print("Tweet: \n")
                print(self.predictTweets_.Text[idx])
                print("\nPrediction: " + prediction)
        numNeg = list(self.predictions_).count('negative')
        numPos = list(self.predictions_).count('positive')
        print("\nRatio of predicted tweets (pos/neg): " + str(numPos) + '/' + str(numNeg))
        if (numPos/(numNeg+numPos)) > 0.75:
            print("Consider investing...")
        elif (numNeg/(numNeg+numPos)) > 0.75:
            print("Don't think you should invest...")
        else:
            print("Too wishy washy... Evaluate more indicators")