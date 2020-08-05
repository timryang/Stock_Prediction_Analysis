# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:21:40 2020

@author: timot
"""


from Stock_Timing_Analyzer import *

#%% Inputs

years = 3
ticker = 'QQQ'
neg_thresh = [0.005, 0.01, 0.015, 0.02]
pos_thresh = [0.005, 0.01, 0.015, 0.02]
sell_adj = [0, 0.25, 0.5, 0.75, 1]
buy_adj = [0, 0.25, 0.5, 0.75, 1]
do_html = False

#%% Execute

timing_analyzer = Stock_Timing_Analyzer()
timing_analyzer.collect_data(ticker, years)
stats_df, results_df, p_hist, p, best_neg_thresh, best_pos_thresh, best_sell_adj, best_buy_adj \
            = Stock_Timing_Grid_Search(timing_analyzer, neg_thresh, pos_thresh,  sell_adj, buy_adj, do_html)
        
print(stats)
print(results_df)
print('Best Neg Thresh: '+str(best_neg_thresh))
print('Best Pos Thresh: '+str(best_pos_thresh))
print('Best Sell Adj: '+str(best_sell_adj))
print('Best Buy Adj: '+str(best_buy_adj))