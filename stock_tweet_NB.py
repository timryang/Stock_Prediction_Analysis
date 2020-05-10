# -*- coding: utf-8 -*-
"""
Created on Sat May  9 19:38:49 2020

@author: Tim
"""
# This script creates a class, stock_NB_Tweet_Analyzer, to collect historical
# tweets used to generate a Naive Bayes machine learning model to predict next
# day's stock performance. An example of using the class is shown in the "main"

#%% Import all necessary libraries
import GetOldTweets3 as got
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter

#%% Functions

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

def get_tweets(txtSearch, startDate = None, stopDate = None, numMaxTweets = 10, topTweets = True):
    # Using open source library (got) to collect tweets based on criteria
    # Returns tweets in DataFrame format consisting of Date, Text, # of Retweets
    if (startDate == None and stopDate == None):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(txtSearch)\
                                                .setMaxTweets(numMaxTweets)\
                                                .setTopTweets(topTweets)
    else:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(txtSearch)\
                                                .setSince(startDate).setUntil(stopDate)\
                                                .setMaxTweets(numMaxTweets)\
                                                .setTopTweets(topTweets)
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
        
    def collect_tweets(self, txtSearch, startDate = None, stopDate = None, numMaxTweets = 10, topTweets = True):
        self.classifierTweets_ = get_tweets(txtSearch, startDate = startDate, stopDate = stopDate, numMaxTweets = numMaxTweets, topTweets = topTweets)

    def plot_tweet_data(self, retweets = True, closeData = True):
        if retweets:
            # Show distribution of tweets by # of retweets
            plt.hist(list(self.classifierTweets_['Retweets']), bins = range(self.classifierTweets_['Retweets'].max()))
            plt.title('Distribution of Retweets')
            plt.xlabel('# of Retweets')
            plt.ylabel('Occurences')
            plt.show()
        if closeData:
            # Plot historical close data over entire dataset
            dates = np.array(self.stockData_['Date'])
            closeData = np.array(self.stockData_['Close'])
            plt.plot(dates, closeData)
            plt.title('Daily Close Performance')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.show()

    def correlate_tweets(self):
        # Correlate each tweet to the next day's stock performance
        for idx in range(len(self.classifierTweets_)):
            tweetDate = self.classifierTweets_['Date'][idx].date()
            tweetTxt = self.classifierTweets_['Text'][idx]
            # Check if the day of tweet and the next day is a weekday
            # (No stock data on weekends)
            if tweetDate.weekday() >= 4:
                # If not, tweet day is correlated to the nearest previous workday
                # And next day is correlated to the nearest next workday
                currentStockDay = tweetDate - datetime.timedelta(days = (tweetDate.weekday() - 4))
                nextStockDay = tweetDate + datetime.timedelta(days = (7 - tweetDate.weekday()))
            else:
                currentStockDay = tweetDate
                nextStockDay = tweetDate + datetime.timedelta(days = 1)
            currentStockDayStr = str(currentStockDay)
            nextStockDayStr = str(nextStockDay)
            # While loop accounts for tweets that fall on a weekday holiday
            # (No stock data on holidays)
            while currentStockDayStr not in (self.stockData_.Date).tolist():
                print("Current date, " + currentStockDayStr + " was not found. Decrementing...")
                currentStockDay = currentStockDay - datetime.timedelta(days = 1)
                currentStockDayStr = str(currentStockDay)
            # Get stock closing price corresponding to the date of the tweet
            currentClose = ((self.stockData_['Close'][self.stockData_.Date == currentStockDayStr]).tolist())[0]
            while nextStockDayStr not in (self.stockData_.Date).tolist():
                print("Next date, " + nextStockDayStr + " was not found. Incrementing...")
                nextStockDay = nextStockDay + datetime.timedelta(days = 1)
                nextStockDayStr = str(nextStockDay)
            # Get stock closing price corresponding to the next day
            nextClose = ((self.stockData_['Close'][self.stockData_.Date == nextStockDayStr]).tolist())[0]
            self.totalTweets_.append(tweetTxt) # Store all tweets in a list
            # Store corresponding stock performance in another list
            if nextClose > currentClose:
                self.totalResults_.append('positive')
            else:
                self.totalResults_.append('negative')       
    
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
        # Normalizes word counts by term frequency within document
        # Takes user input (useIDF) to decide whether to account inverse document frequency
        self.tfTransformer_ = TfidfTransformer(use_idf = useIDF)
        train_tf = self.tfTransformer_.fit_transform(train_counts)
        # Create multinomial Naive Bayes model (binomial in this case) using historical data
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
        # Print total tweets occurring in each label
        print("\n" + classes[0] + " total: " + str(self.clf_.class_count_[0]))
        print(classes[1] + " total: " + str(self.clf_.class_count_[1]))
        # Print dictionary with ratio results - this indicates the most "informative" words
        self.ratio_dict_ = {label_one2two: top_one2two, label_two2one: top_two2one}
        print(self.ratio_dict_)

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
        print("Ratio of negative to positve: " + str(numNeg) + '/' + str(numPos))
    
    
#%% Main Code As Example
    
if __name__ == "__main__":
    
    #%% Create Analyzer
    # Load stock history in csv format. This can be found on yahoo finance webpage
    stockData = pd.read_csv('./CSV_Files/TSLA.csv')
    tslaAnalyzer = stock_NB_Tweet_Analyzer(stockData)
    
    #%% Collect Historical Tweets
    # Classifier Tweet Criteria
    classifyTxtSearch = 'TSLA'
    classifyNumMaxTweets = 0
    classifyStartDate = '2020-01-01'
    classifyStopDate = '2020-04-30'
    classifyTopTweets = True
    
    tslaAnalyzer.collect_tweets(classifyTxtSearch, classifyStartDate, classifyStopDate,\
                                classifyNumMaxTweets, classifyTopTweets)
    
    #%% Correlate Historical Tweets to Stock Performance
    tslaAnalyzer.correlate_tweets()
    
    #%% Create Classifier to Predict Tweets
    trainSize = 0.8
    stopwordsList = stopwords.words('english')
    useIDF = True
    
    tslaAnalyzer.create_classifier(trainSize, stopwordsList, useIDF)
    
    #%% Analyze Classifier Data
    plotRetweets = True
    plotCloseData = True
    tslaAnalyzer.plot_tweet_data(retweets = plotRetweets, closeData = plotCloseData)
    
    numFeatures = 10
    tslaAnalyzer.show_most_informative(n = numFeatures)
    
    #%% Create Prediction from New Tweets
    # Prediction Tweet Criteria
    predictTxtSearch = 'TSLA'
    predictNumMaxTweets = 10
    predictTopTweets = True
    printAll = True
    
    tslaAnalyzer.predict(predictTxtSearch, predictNumMaxTweets, predictTopTweets, printAll)