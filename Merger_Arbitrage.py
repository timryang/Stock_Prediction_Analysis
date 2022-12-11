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
        self.announceDate_ = ''
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
        self.userName_ = ''
        self.searchString_ = ''
        self.maxTweets_ = []
        self.language_ = ''
        self.probThresh_ = []
        self.dayGuard_ = []
        self.mostInform_ = pd.DataFrame()
        self.fig1_ = go.Figure()
        self.fig2_ = go.Figure()
        self.tweetString_ = ''
        
        
    def collect_data(self, analyzeTicker, refTicker, announceDate):
        self.analyzeTicker_ = analyzeTicker
        self.refTicker_ = refTicker
        self.announceDate_ = announceDate
        
        # Get previous business day
        announceDatetime = datetime.datetime.strptime(announceDate, '%m-%d-%Y')
        startDate = datetime.datetime.strftime(prev_business_day(announceDatetime), '%m-%d-%Y')
        
        self.analyzeStockData_ = collect_stock_data(analyzeTicker, startDate)
        self.refStockData_ = collect_stock_data(refTicker, startDate)
        self.dates_ = pd.to_datetime(self.analyzeStockData_['Date']).tolist()
        
    def find_probability(self, acqPrice):
        
        refCloseData = self.refStockData_['Close'].values
        analyzeCloseData = self.analyzeStockData_['Close'].values
        
        refPercentDifference = refCloseData/refCloseData[0]
        fallPrice = analyzeCloseData[0]*refPercentDifference
        self.percFall_ = (analyzeCloseData-fallPrice)/analyzeCloseData
        
        self.probAcq_ = np.divide(analyzeCloseData-fallPrice, acqPrice-fallPrice)
        
        self.fallPrice_ = fallPrice
        self.acqPrice_ = acqPrice
        
    def collectTweets(self, searchString, probThresh, dayGuard, maxTweets, userName=None, lang='en'):
        probDiff = np.diff(self.probAcq_)[1::]*100
        stockDates = self.dates_[2::]
        
        annotateTxt = []
        annotateDates = []
        
        gainIdx = np.where(probDiff>probThresh,True,False)
        gainDates = np.array(stockDates)[gainIdx]
        gainTweets = []
        for i,iterDate in enumerate(gainDates):
            stopDate = iterDate+datetime.timedelta(days=dayGuard-1)
            tempTweets = get_tweets_list(searchString,iterDate.strftime('%m-%d-%Y'),stopDate.strftime('%m-%d-%Y'),maxTweets,userName,lang)
            gainTweets.extend(tempTweets)
            tweetScore = np.array([tweet[4]+tweet[5] for tweet in tempTweets])
            bestTweet = tempTweets[np.argmax(tweetScore)]
            annotateDates.append(iterDate)
            annotateTxt.append(bestTweet[2])
        
        self.allGainTxt_ = [tweet[2] for tweet in gainTweets]
        
        lossIdx = np.where(probDiff<-probThresh,True,False)
        lossDates = np.array(stockDates)[lossIdx]
        lossTweets = []
        for i,iterDate in enumerate(lossDates):
            stopDate = iterDate+datetime.timedelta(days=dayGuard-1)
            tempTweets = get_tweets_list(self.analyzeTicker_,iterDate.strftime('%m-%d-%Y'),stopDate.strftime('%m-%d-%Y'),maxTweets,userName,lang)
            lossTweets.extend(tempTweets)
            tweetScore = np.array([tweet[4]+tweet[5] for tweet in tempTweets])
            bestTweet = tempTweets[np.argmax(tweetScore)]
            annotateDates.append(iterDate)
            annotateTxt.append(bestTweet[2])
            
        self.allLossTxt_ = [tweet[2] for tweet in lossTweets]
        
        sortIdx = np.argsort(annotateDates)
        self.annotateTxt_ = [annotateTxt[i] for i in sortIdx]
        self.annotateDates_ = [annotateDates[i] for i in sortIdx]
        
        self.searchString_ = searchString
        self.probThresh_ = probThresh
        self.dayGuard_ = dayGuard
        self.maxTweets_ = maxTweets
        self.userName_ = userName
        self.language_ = lang
        
    def requeryTweets(self, searchString, searchDate, maxTweets, userName=None, lang='en'):
        searchDatetime = datetime.datetime.strptime(searchDate, '%m-%d-%Y')
        tempTweets = get_tweets_list(searchString,searchDatetime.strftime('%m-%d-%Y'),searchDatetime.strftime('%m-%d-%Y'),100,userName,lang)
        tweetScore = np.array([tweet[4]+tweet[5] for tweet in tempTweets])
        sortIndices = np.flip(np.argsort(tweetScore))
        if len(sortIndices) < maxTweets:
            maxTweets = len(sortIndices)
        topIndices = sortIndices[:maxTweets]
        requeryString = ''
        for idx in topIndices:
            requeryString += str(tempTweets[idx][0].strftime('%m-%d-%Y'))+': '+tempTweets[idx][2]+'\n\n'
        return requeryString
        
    def evalDictionStats(self, trainSize=0.9, stopwordsList=stopwords.words('english'), useIDF=True, doHTML=True):
        x_values = self.allGainTxt_ + self.allLossTxt_
        y_values = ['Pos']*len(self.allGainTxt_)+['Neg']*len(self.allLossTxt_)
        clf, count_vect, tfTransformer, report, most_inform, p = create_NB_text_classifier(x_values, y_values, trainSize, stopwordsList, useIDF, doHTML=doHTML)
        self.mostInform_ = most_inform
        return most_inform
        
    def plot_data(self, doHTML=False):
        
        # First Plot
        closeDataTrace = go.Scatter(x=self.dates_, y=self.analyzeStockData_['Close'].values, name='Actual', yaxis='y1')
        fallDataTrace = go.Scatter(x=self.dates_, y=self.fallPrice_, name='Market', yaxis='y1')
        acqPriceTrace = go.Scatter(x=self.dates_, y=self.acqPrice_*np.ones(np.size(self.fallPrice_)), name='Acq. Price', yaxis='y1')
        probAcqTrace = go.Scatter(x=self.dates_, y=self.probAcq_*100, name='Acq. Prob', line={'dash': 'dash'}, yaxis='y2')
        percFallTrace = go.Scatter(x=self.dates_, y=self.percFall_*100, name='Fall Perc.', line={'dash': 'dash'}, yaxis='y2')
        layout = go.Layout(title=self.analyzeTicker_, yaxis=dict(title='Price'), yaxis2=dict(title='Percentage',\
                           overlaying='y', side='right'), xaxis=dict(title='Date'),\
                           legend={'orientation': 'h', 'y': -0.2}, showlegend=True)
        fig1 = go.Figure(data=[closeDataTrace,fallDataTrace,acqPriceTrace,probAcqTrace,percFallTrace], layout=layout)
        
        if not doHTML:
            plot(fig1)
        
        # Second plot
        tweetString = ''
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=self.dates_[1::], y=self.probAcq_[1::]*100, name='Acq. Prob'))
        for iTxt,iDate in enumerate(self.annotateDates_):
            yVal = self.probAcq_[np.where(np.array(self.dates_)==iDate)[0][0]]*100
            xTxt = self.annotateDates_[iTxt]+datetime.timedelta(days=5)
            if  (iTxt%2==0):
                yTxt = yVal-5
            else:
                yTxt = yVal+5
            fig2.add_annotation(x=self.annotateDates_[iTxt], y=yVal, text=str(iTxt), showarrow=True, arrowhead=1)
            tweetString += '['+str(iTxt)+'] '+str(iDate.strftime('%m-%d-%Y'))+' : '+self.annotateTxt_[iTxt]+'\n\n'
        fig2.update_layout(title='Probabiliy Acquisition', xaxis_title='Date', yaxis_title='Prob (%)', showlegend=False)
        
        if not doHTML:
            plot(fig2)
            print(tweetString)
        
        self.fig1_ = fig1
        self.fig2_ = fig2
        self.tweetString_ = tweetString
        
        return fig1, fig2, tweetString