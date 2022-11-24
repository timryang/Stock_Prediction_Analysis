# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 11:14:10 2022

@author: timot
"""


from Merger_Arbitrage import *
    

#%% Inputs

# Stock data
analyzeTicker = 'ATVI'
refTicker = 'ESPO'

# Acquisition details
announceDate = '01-18-2022'
acqPrice = 95

# Twitter details
probThresh = 2.5
dayGuard = 1
maxTweets = 25
searchString = 'ATVI AND MSFT'

#%% Execute

merger_analyzer = Merger_Arbitrage()
merger_analyzer.collect_data(analyzeTicker,refTicker,announceDate)
merger_analyzer.find_probability(acqPrice)
merger_analyzer.collectTweets(searchString, probThresh, dayGuard, maxTweets)
merger_analyzer.evalDictionStats()
merger_analyzer.plot_data()

#%% Testing

# lossQuery = 'callofduty'
# for i,txt in enumerate(merger_analyzer.lossTxt_):
#     if lossQuery.upper() in txt.upper():
#         print(txt+'\n')
        
# gainQuery = 'va9jrbtnic'
# for i,txt in enumerate(merger_analyzer.gainTxt_):
#     if gainQuery.upper() in txt.upper():
#         print(txt+'\n')