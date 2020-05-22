# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:04:37 2020

@author: timot
"""
from commonFunctions import *

#%%  Functions

class Stock_NB_Analyzer:
    
    def __init__(self):
        self.stockData_ = pd.DataFrame()
        self.tweetsDF_ = pd.DataFrame()
        self.tweetResults_ = []
        self.countVect_ = CountVectorizer()
        self.tfTransformer_ = TfidfTransformer()
        self.clf_ = MultinomialNB()
        
    def collect_tweets(self, geoLocation, distance, sinceDate, untilDate, querySearch, maxTweets, topTweets):
        self.tweetsDF_ = get_tweets(querySearch, startDate = sinceDate, stopDate = untilDate,\
                          geoLocation = geoLocation, distance = distance,\
                              topTweets = topTweets, numMaxTweets = maxTweets)
    
    def load_tweets(self, directory):
        self.tweetsDF_ = pd.read_csv(directory)
        
    def collect_data(self, ticker, years):
        oneYearUnix = 31536000
        tempLink = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker\
            + "?period1=" + str(int(time.time())-(oneYearUnix*years)) + "&period2="\
                + str(int(time.time())) + "&interval=1d&events=history"
        try:
            stockData = pd.read_csv(tempLink)
        except:
            raise ValueError("Bad link: Double check your ticker")
        self.stockData_ = stockData
        
    def load_data(self, directory):
        self.stockData_ = pd.read_csv(directory)
        
    def correlate_tweets(self, deltaInterval):
        
        stockClose = self.stockData_['Close'].values
        diffInterval = [stockClose[i+deltaInterval]-val for i, val in enumerate(stockClose[:-deltaInterval])]
        stockDates = list(self.stockData_['Date'])
        stockDates = stockDates[:-deltaInterval]
        
        resultsClose = np.where(np.array(diffInterval) > 0, 'positive', 'negative')
        
        lastDate = pd.to_datetime(stockDates[-1])
        tweetDatesAll = [dt.date() for dt in pd.to_datetime(self.tweetsDF_['Date'])]
        validIdx = [idx for idx, val in enumerate(tweetDatesAll) if val <= lastDate]
        tweetsDF_short = self.tweetsDF_.iloc[validIdx]
        tweetDates_short = [dt.date() for dt in pd.to_datetime(tweetsDF_short['Date'])]
        
        tweetResults = []
        for idx, tweetDate in enumerate(tweetDates_short):
            if tweetDate.weekday() >= 4:
                currentStockDay = tweetDate - datetime.timedelta(days = (tweetDate.weekday() - 4))
            else:
                currentStockDay = tweetDate
            currentStockDayStr = str(currentStockDay)
            while currentStockDayStr not in stockDates:
                currentStockDay = currentStockDay - datetime.timedelta(days = 1)
                currentStockDayStr = str(currentStockDay)
            result = resultsClose[find_idx(stockDates, currentStockDayStr)]
            tweetResults.append(result[0])
            
        numPosDays = list(resultsClose).count('positive')
        numNegDays = list(resultsClose).count('negative')
        print("\nTotal positive days: " + str(numPosDays))
        print("Total negative days: " + str(numNegDays))
        print("Percent positive days: %0.2f" % (numPosDays/(numNegDays + numPosDays)*100))
        
        numPosTweets = tweetResults.count('positive')
        numNegTweets = tweetResults.count('negative')
        print("\nTotal positive tweets: " + str(numPosTweets))
        print("Total negative tweets: " + str(numNegTweets))
        print("Percent positive tweets: %0.2f" % (numPosTweets/(numNegTweets + numPosTweets)*100))
        
        self.tweetResults_ = tweetResults
        self.tweetsDF_ = tweetsDF_short
        
        return tweetResults, tweetsDF_short
        
    def create_classifier(self, trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures):
        totalTweets = list(self.tweetsDF_['Text'])
        self.clf_, self.count_vect_, self.tfTransformer_\
        = create_NB_text_classifier(totalTweets, self.tweetResults_, trainSize, stopwordsList,\
                                    useIDF, do_downsample=do_downsample,
                                    do_stat=do_stat, n_features=numFeatures)
            
    def run_prediction(self, geoLocation, distance, txtSearch, numMaxTweets, topTweets, printAll):
        predict_from_tweets(self.clf_, self.count_vect_, self.tfTransformer_,\
            txtSearch, geoLocation, distance, numMaxTweets,\
                topTweets, printAll)
    
    def plot_data(self, deltaInterval):
        dates = pd.to_datetime(self.stockData_['Date'])
        
        plt.hist(list(self.tweetsDF_['Retweets']), bins = range(self.tweetsDF_['Retweets'].max()))
        plt.title('Distribution of Retweets')
        plt.xlabel('# of Retweets')
        plt.ylabel('Occurences')
        plt.show()
        
        stockClose = self.stockData_['Close'].values
        
        diffClose = np.diff(stockClose)
        diffDates = dates[:-1]
        diffInterval = [stockClose[i+deltaInterval]-val for i, val in enumerate(stockClose[:-deltaInterval])]
        diffIntDates = dates[:-deltaInterval]
        
        x_values_plt1 = [dates]
        y_values_plt1 = [stockClose]
        labels_plt1 = ['Close Data']
        title_plt1 = 'Daily Close'
        xlabel_plt1 = 'Date'
        ylabel_plt1 = 'Price'
        
        x_values_plt2 = [diffDates, diffIntDates]
        y_values_plt2 = [diffClose, diffInterval]
        labels_plt2 = ['Daily', 'Interval']
        title_plt2 = 'Change By Interval'
        xlabel_plt2 = 'Date'
        ylabel_plt2 = 'Change'
        
        plot_values(x_values_plt1, y_values_plt1, labels_plt1, xlabel_plt1, ylabel_plt1, title_plt1, isDates=True)
        plot_values(x_values_plt2, y_values_plt2, labels_plt2, xlabel_plt2, ylabel_plt2, title_plt2, isDates=True)