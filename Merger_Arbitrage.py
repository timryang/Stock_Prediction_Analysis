# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:00:09 2022

@author: timot
"""


from CommonFunctions.commonFunctions import *

#%% Class definition

class Merger_Arbitrage():
    
    def __init__(self):
        self.analyzeTicker_ = ''
        self.refTicker_ = ''
        self.analyzeStockData_ = pd.DataFrame()
        self.refStockData_ = pd.DataFrame()
        self.acqPrice_ = []
        self.probAcq_ = np.array([])
        self.fallPrice_ = np.array([])
        self.percFall_ = np.array([])
        self.dates_ = np.array([])
        self.allGainTxt_ = [];
        self.allLossTxt_ = [];
        self.annotateDates_ = [];
        self.annotateTxt_ = [];
        
    def collect_data(self, analyzeTicker, refTicker, announceDate):
        self.analyzeTicker_ = analyzeTicker
        self.refTicker_ = refTicker
        
        # Get previous business day
        announceDatetime = datetime.datetime.strptime(announceDate, '%m-%d-%Y')
        startDate = datetime.datetime.strftime(prev_business_day(announceDatetime), '%m-%d-%Y')
        
        self.analyzeStockData_ = collect_stock_data(analyzeTicker, startDate)
        self.refStockData_ = collect_stock_data(refTicker, startDate)
        self.dates_ = pd.to_datetime(self.analyzeStockData_['Date'])
        
    def find_probability(self, acqPrice):
        
        refCloseData = self.refStockData_['Close'].values
        analyzeCloseData = self.analyzeStockData_['Close'].values
        
        refPercentDifference = refCloseData/refCloseData[0]
        fallPrice = analyzeCloseData[0]*refPercentDifference
        self.percFall_ = (analyzeCloseData-fallPrice)/analyzeCloseData
        
        self.probAcq_ = np.divide(analyzeCloseData-fallPrice, acqPrice-fallPrice)
        
        self.fallPrice_ = fallPrice
        self.acqPrice_ = acqPrice
        
    def collectTweets(self, searchString, probThresh, dayGuard, maxTweets):
        probDiff = np.diff(self.probAcq_)[1::]*100
        stockDates = self.dates_[2::]
        
        annotateTxt = []
        annotateDates = []
        
        gainIdx = np.where(probDiff>probThresh,True,False)
        gainDates = stockDates[gainIdx]
        gainTweets = []
        for i,iterDate in enumerate(gainDates):
            stopDate = iterDate+datetime.timedelta(days=dayGuard-1)
            tempTweets = get_tweets_list(searchString,iterDate.strftime('%m-%d-%Y'),stopDate.strftime('%m-%d-%Y'),maxTweets)
            gainTweets.extend(tempTweets)
            tweetScore = np.array([tweet[4]+tweet[5] for tweet in tempTweets])
            bestTweet = tempTweets[np.argmax(tweetScore)]
            annotateDates.append(iterDate)
            annotateTxt.append(bestTweet[2])
        
        self.allGainTxt_ = [tweet[2] for tweet in gainTweets]
        
        lossIdx = np.where(probDiff<-probThresh,True,False)
        lossDates = stockDates[lossIdx]
        lossTweets = []
        for i,iterDate in enumerate(lossDates):
            stopDate = iterDate+datetime.timedelta(days=dayGuard-1)
            tempTweets = get_tweets_list(self.analyzeTicker_,iterDate.strftime('%m-%d-%Y'),stopDate.strftime('%m-%d-%Y'),maxTweets)
            lossTweets.extend(tempTweets)
            tweetScore = np.array([tweet[4]+tweet[5] for tweet in tempTweets])
            bestTweet = tempTweets[np.argmax(tweetScore)]
            annotateDates.append(iterDate)
            annotateTxt.append(bestTweet[2])
            
        self.allLossTxt_ = [tweet[2] for tweet in lossTweets]
        
        sortIdx = np.argsort(annotateDates)
        self.annotateTxt_ = [annotateTxt[i] for i in sortIdx]
        self.annotateDates_ = [annotateDates[i] for i in sortIdx]
        
    def evalDictionStats(self, trainSize=0.9, stopwordsList=stopwords.words('english'), useIDF=True):
        x_values = self.allGainTxt_ + self.allLossTxt_
        y_values = ['Pos']*len(self.allGainTxt_)+['Neg']*len(self.allLossTxt_)
        clf, count_vect, tfTransformer, report, most_inform, p = create_NB_text_classifier(x_values, y_values, trainSize, stopwordsList, useIDF)
        
    def plot_data(self):
        
        # First Plot
        x_values = [self.dates_, self.dates_, self.dates_]
        y_values = [self.analyzeStockData_['Close'].values, self.fallPrice_, self.acqPrice_*np.ones(np.size(self.fallPrice_))]
        labels = ['Actual', 'Market', 'Acq. Price']
        
        lines = np.tile(np.array(['-']),len(labels)+1)
        plt.figure(figsize=(20,10))
        formatter = mdates.DateFormatter("%m-%d-%Y")
        locator = mdates.DayLocator(bymonthday=[1, 15])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        plt.xticks(rotation = 70)
        ax.set_title(self.analyzeTicker_)
        ax.set_xlabel('Date')
        
        for idx, i_y_values in enumerate(y_values):
            ax.plot_date(x_values[idx], i_y_values, lines[idx], label=labels[idx])
        handles, labels_return = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_return, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set_ylabel('Price')
        
        ax2 = ax.twinx()
        ax2.plot_date(self.dates_, self.probAcq_*100, 'k--', label = 'Acq. Prob')
        ax2.plot_date(self.dates_, self.percFall_*100, 'r--', label = 'Fall Perc.')
        ax2.legend(['Acq. Prob', 'Fall Perc.'], loc = 'lower center')
        ax2.set_ylabel('Prob (%)')
        
        plt.grid()
        plt.show()
        
        # Second plot
        plt.figure(figsize=(20,10))
        formatter = mdates.DateFormatter("%m-%d-%Y")
        locator = mdates.DayLocator(bymonthday=[1, 15])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(locator)
        plt.xticks(rotation = 70)
        ax.set_title('Probability Acquisition')
        ax.set_xlabel('Date')
        ax.plot_date(self.dates_, self.probAcq_*100, 'k-', label='Acq. Prob')
        ax.set_ylabel('Prob (%)')
        ax.set_ylim([np.min([0,np.min(self.probAcq_*100)-10]), np.max(self.probAcq_*100)+10])
        
        for iTxt,iDate in enumerate(self.annotateDates_):
            yVal = self.probAcq_[np.where(np.array([self.dates_])[0]==iDate)[0][0]]*100
            xTxt = self.annotateDates_[iTxt]+datetime.timedelta(days=5)
            if  (iTxt%2==0):
                yTxt = yVal-5
            else:
                yTxt = yVal+5
            ax.annotate(str(iTxt), xy = (self.annotateDates_[iTxt], yVal),\
                 fontsize = 20, xytext = (xTxt, yTxt),\
                 arrowprops = dict(facecolor = 'red'),\
                 color = 'g')
            print('['+str(iTxt)+'] '+str(iDate.strftime('%m-%d-%Y'))+' : '+self.annotateTxt_[iTxt]+'\n')
        
        plt.grid()
        plt.show()