# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:04:37 2020

@author: timot
"""
from CommonFunctions.commonFunctions import *

#%%  Functions

class Stock_NB_Analyzer:
    
    def __init__(self):
        self.stockData_ = pd.DataFrame()
        self.tweetsDF_ = pd.DataFrame()
        self.tweetResults_ = []
        self.countVect_ = CountVectorizer()
        self.tfTransformer_ = TfidfTransformer()
        self.clf_ = MultinomialNB()
        
    def collect_tweets(self, userName, sinceDate, untilDate, querySearch, maxTweets, lang='en'):
        self.tweetsDF_ = get_tweets(querySearch, startDate=sinceDate, stopDate=untilDate,\
                                    maxTweets=maxTweets, userName=userName, lang=lang)
    
    def load_tweets(self, directory):
        if isinstance(directory, pd.DataFrame):
            self.tweetsDF_ = directory
        else:
            self.tweetsDF_ = pd.read_csv(directory)
        
    def collect_data(self, ticker, startDate):
        stockData = collect_stock_data(ticker, startDate)
        self.stockData_ = stockData
        
    def load_data(self, directory):
        if isinstance(directory, pd.DataFrame):
            self.stockData_ = directory
        else:
            self.stockData_ = pd.read_csv(directory)
        
    def correlate_tweets(self, deltaInterval, changeFilter=0):
        
        stockClose = self.stockData_['Close'].values
        diffInterval = np.array([stockClose[i+deltaInterval]-val for i, val in enumerate(stockClose[:-deltaInterval])])
        diffInterval = diffInterval/stockClose[:-deltaInterval]
        stockDates = [dt.date() for dt in pd.to_datetime(self.stockData_['Date'].values)]
        stockDates = stockDates[:-deltaInterval]
        
        resultsClose = np.where(diffInterval > 0, 'Positive', 'Negative')
        validIdx = np.where(np.abs(diffInterval) > changeFilter)
        
        tweetDatesAll = np.array([dt.date() for dt in pd.to_datetime(self.tweetsDF_['Date'])])
        tweetsDF_short = self.tweetsDF_.iloc[np.where(tweetDatesAll <= stockDates[-1])]
        tweetDates_short = [dt.date() for dt in pd.to_datetime(tweetsDF_short['Date'])]
        
        tweetResults = []
        for idx, tweetDate in enumerate(tweetDates_short):
            if (tweetDate.weekday() in holidays.WEEKEND or tweetDate in holidays.US()):
                currentStockDay = prev_business_day(tweetDate)
            else:
                currentStockDay = tweetDate
            while (currentStockDay not in stockDates and (currentStockDay-stockDates[0]).days>=0):
                currentStockDay = currentStockDay - datetime.timedelta(days = 1)
            
            matchIdx = np.where(np.array(stockDates)==currentStockDay)[0]
            if (len(matchIdx)>0 and matchIdx[0] in validIdx[0]):
                tweetResults.append(resultsClose[matchIdx][0])
            else:
                tweetResults.append('Non-valid')
        
        validTweetsIdx = np.where(np.array(tweetResults) != 'Non-valid')
        tweetResults = list(np.array(tweetResults)[validTweetsIdx])
        tweetsDF_short = tweetsDF_short.iloc[validTweetsIdx]
        
        numPosDays = list(resultsClose[validIdx]).count('Positive')
        numNegDays = list(resultsClose[validIdx]).count('Negative')
        
        numPosTweets = tweetResults.count('Positive')
        numNegTweets = tweetResults.count('Negative')
        
        count_report = "Total Positive Days: " + str(numPosDays) +\
            "\nTotal Negative Days: " + str(numNegDays) +\
                "\nPerc Positive Days: %0.2f" % (numPosDays/(numNegDays + numPosDays+0.001)*100) +\
                    "\n\nTotal Positive Tweets: " + str(numPosTweets) +\
                        "\nTotal Negative Tweets: " + str(numNegTweets) +\
                            "\nPerc Positive Tweets: %0.2f" % (numPosTweets/(numNegTweets + numPosTweets+0.001)*100)
        print(count_report)
        
        self.tweetResults_ = tweetResults
        self.tweetsFilt_ = tweetsDF_short
        
        return count_report
        
    def create_classifier(self, trainSize, stopwordsList, useIDF, do_downsample, do_stat, numFeatures, doHTML=False):
        totalTweets = list(self.tweetsFilt_['Text'])
        self.clf_, self.count_vect_, self.tfTransformer_, report, most_inform, p\
        = create_NB_text_classifier(totalTweets, self.tweetResults_, trainSize, stopwordsList,\
                                    useIDF, do_downsample=do_downsample,
                                    do_stat=do_stat, n_features=numFeatures, doHTML=doHTML)
        return report, most_inform, p
            
    def run_prediction(self, userName, txtSearch, numMaxTweets, lang, printAll):
        results, predictionTxt = predict_from_tweets(self.clf_, self.count_vect_, self.tfTransformer_,\
            txtSearch, userName, numMaxTweets, printAll)
        return results, predictionTxt
    
    def plot_data(self, deltaInterval, doHTML=False):
        
        unique_tweet_dates, num_daily_retweets = count_retweets_by_day(self.tweetsDF_)
        unique_tweet_dates = pd.to_datetime(unique_tweet_dates)
        
        dates = pd.to_datetime(self.stockData_['Date'])
        
        p_hist = go.Figure()
        p_hist.add_trace(go.Histogram(x=self.tweetsDF_['Retweets'].values))
        p_hist.update_layout(title='Distribution of Retweets', xaxis_title='# of Retweets', yaxis_title='Occurrences')
        if not doHTML:
            plot(p_hist)
        
        stockClose = self.stockData_['Close'].values
        
        closeDataTrace = go.Scatter(x=dates, y=stockClose, name='Close Data', yaxis='y1')
        twitterCountTrace = go.Scatter(x=unique_tweet_dates, y=num_daily_retweets, name='Retweets', yaxis='y2')
        layout = go.Layout(title='Daily Close and Num Reweets', yaxis=dict(title='Close'), yaxis2=dict(title='Retweets',\
                           overlaying='y', side='right'), xaxis=dict(title='Date'))
        p = go.Figure(data=[closeDataTrace,twitterCountTrace], layout=layout)
        if not doHTML:
            plot(p)
        
        diffClose = np.diff(stockClose)
        diffClose = diffClose/stockClose[:-1]
        diffDates = dates[:-1]
        diffInterval = np.array([stockClose[i+deltaInterval]-val for i, val in enumerate(stockClose[:-deltaInterval])])
        diffInterval = diffInterval/stockClose[:-deltaInterval]
        diffIntDates = dates[:-deltaInterval]
        
        p_delta = go.Figure()
        p_delta.add_trace(go.Scatter(x=diffDates, y=diffClose, name='Daily'))
        p_delta.add_trace(go.Scatter(x=diffIntDates, y=diffInterval, name='Interval'))
        p_delta.update_layout(title='Change By Interval', xaxis_title='Date', yaxis_title='% Change')
        
        x_values_plt2 = [diffDates, diffIntDates]
        y_values_plt2 = [diffClose, diffInterval]
        labels_plt2 = ['Daily', 'Interval']
        title_plt2 = 'Change By Interval'
        xlabel_plt2 = 'Date'
        ylabel_plt2 = '% Change'
        if not doHTML:
            plot(p_delta)
        
        return p
    
def Stock_NB_Grid_Search(NB_analyzer, trainSize, deltaInterval, changeFilter, useIDF, do_downsample, stopwordsList, do_stat=True, numFeatures=10, doHTML=True):
        
    best_score = -1000
    
    for i_int in deltaInterval:
        for i_change in changeFilter:
            count_report = NB_analyzer.correlate_tweets(i_int, i_change)
            p = NB_analyzer.plot_data(i_int, doHTML=doHTML)
            for i_train in trainSize:
                for i_idf in useIDF:
                    for i_ds in do_downsample:
                        for i_sw in stopwordsList:
                            report, most_inform, conf_mat = NB_analyzer.create_classifier(i_train, i_sw, i_idf, i_ds, do_stat=do_stat, numFeatures=numFeatures, doHTML=doHTML)
                            
                            report_score = report['F1'].iloc[np.where(report['Class/Metric'].values=='accuracy')[0][0]]
                            
                            if (report_score > best_score):
                                best_score = report_score
                                best_NB = NB_analyzer
                                best_cr = count_report
                                best_p = p
                                best_report = report
                                best_most_inform = most_inform
                                best_conf_mat = conf_mat
                                best_int = i_int
                                best_change = i_change
                                best_train = i_train
                                best_idf = i_idf
                                best_ds = i_ds
                                best_sw = i_sw
                        
    return best_NB, best_cr, best_p, best_report, best_most_inform, best_conf_mat, best_int, best_change, best_train, best_idf, best_ds, best_sw