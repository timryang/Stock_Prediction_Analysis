# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:50:22 2020

@author: timot
"""


from CommonFunctions.commonFunctions import *

#%% Stock timing class

class Stock_Timing_Analyzer():
    
    def __init__(self):
        self.analyzeTicker_ = ''
        self.stock_data_ = pd.DataFrame()
        self.stock_close = np.array([])
        self.stock_perc_change = np.array([])
        self.statistics_ = pd.DataFrame()
        self.change_before_neg_ = np.array([])
        self.neg_intervals_ = np.array([])
        self.change_before_pos_ = np.array([])
        self.pos_intervals_ = np.array([])
        self.sell_trigger_ = 0
        self.buy_trigger_ = 0
        self.test_val_ = np.array([])
        self.cum_change_ = []
        self.metric_ = 1
        
    def collect_data(self, analyzeTicker, years):
        self.analyzeTicker_ = analyzeTicker
        self.stock_data_ = collect_stock_data(analyzeTicker, years)
        
    def sort_data(self, neg_thresh, pos_thresh):
        stock_close = self.stock_data_['Close'].values
        stock_change = np.diff(stock_close)
        stock_perc_change = stock_change/stock_close[:-1]
        
        self.stock_close_ = stock_close
        self.stock_perc_change_ = stock_perc_change
        
        neg_indices = np.where(stock_perc_change <= neg_thresh*-1)[0]
        self.change_before_neg_ = stock_perc_change[neg_indices-1]
        self.neg_intervals_ = np.diff(neg_indices)
        
        pos_indices = np.where(stock_perc_change >= pos_thresh)[0]
        self.change_before_pos_ = stock_perc_change[pos_indices-1]
        self.pos_intervals_ = np.diff(pos_indices)
    
    def do_stat(self, sell_adj, buy_adj):
        stat_df = {'% Before Drop':pd.Series(self.change_before_neg_),'Days Between Drop':pd.Series(self.neg_intervals_), \
           '% Before Increase':pd.Series(self.change_before_pos_),'Days Between Increase':pd.Series(self.pos_intervals_)}
        stat_df = pd.DataFrame(stat_df)
        self.statistics_ = (stat_df.describe()).round(3)
        self.sell_trigger_ = np.mean(self.change_before_neg_)+(np.std(self.change_before_neg_)*sell_adj)
        self.buy_trigger_ = np.mean(self.change_before_pos_)-(np.std(self.change_before_pos_)*buy_adj)
        return self.statistics_, self.sell_trigger_, self.buy_trigger_
    
    def do_test(self):
        test_val =  np.zeros(np.shape(self.stock_close_))
        test_val[0] = 1
        own = 1
        for idx, i_change in enumerate(self.stock_perc_change_):
            if own == 1:
                test_val[idx+1] = test_val[idx]*(1+i_change)
                if i_change >= self.sell_trigger_:
                    own = 0
            elif own == 0:
                test_val[idx+1] = test_val[idx]
                if i_change <= self.buy_trigger_:
                    own = 1
        
        cum_test_change = test_val/test_val[0]
        cum_real_change = self.stock_close_/self.stock_close_[0]
        
        self.cum_change_ = [cum_test_change, cum_real_change]
        self.metric_ = cum_test_change[-1]-cum_real_change[-1]
        self.test_val_ = test_val
        return self.metric_
        
    def plot_data(self, doHTML):
        p_hist_neg = plot_hist(values=self.change_before_neg_, bin_interval=0.005, x_label='% Change', y_label='Occurrences', \
                       title='% Change Prior to Drop', isBokeh=doHTML)
        p_hist_neg_int = plot_hist(values=self.neg_intervals_, bin_interval=1, x_label='Days', y_label='Occurrences', \
                               title='Days Between Drops', isBokeh=doHTML)
        p_hist_pos = plot_hist(values=self.change_before_pos_, bin_interval=0.005, x_label='% Change', y_label='Occurrences', \
                       title='% Change Prior to Increase', isBokeh=doHTML)
        p_hist_pos_int = plot_hist(values=self.pos_intervals_, bin_interval=1, x_label='Days', y_label='Occurrences', \
                               title='Days Between Increases', isBokeh=doHTML)
        
        stock_dates = pd.to_datetime(self.stock_data_['Date'].values)
        dates_change = stock_dates[1::]
        labels = ['Test', self.analyzeTicker_]
        p_close_perc = plot_values(x_values=[stock_dates, stock_dates], y_values=self.cum_change_, labels=labels, \
                            x_label='Dates', y_label='Cumualtive %', title='Cumulative % Change', \
                                isDates=True, isBokeh=doHTML)
            
        test_val_change = np.diff(self.test_val_)/self.test_val_[:-1]
        stock_close_change = np.diff(self.stock_close_)/self.stock_close_[:-1]
        close_change = [test_val_change, stock_close_change]
        p_close_change = plot_values(x_values=[dates_change, dates_change], y_values=close_change, labels=labels, \
                            x_label='Dates', y_label='% Change', title='Daily % Change', \
                                isDates=True, isBokeh=doHTML)
        
        if doHTML:
            p_hist_change = bokeh.layouts.row(p_hist_neg, p_hist_pos, sizing_mode='stretch_both')
            p_hist_int = bokeh.layouts.row(p_hist_neg_int, p_hist_pos_int, sizing_mode='stretch_both')
            p_hist = bokeh.layouts.row(p_hist_change, p_hist_int, sizing_mode='stretch_both')
            p = bokeh.layouts.row(p_close_perc, p_close_change, sizing_mode='stretch_both')
        else:
            p_hist = 'IDE'
            p = 'IDE'
        
        return p_hist, p
    
def Stock_Timing_Grid_Search(timing_analyzer, neg_thresh, pos_thresh, sell_adj, buy_adj, doHTML):
    
    best_score = -1
    
    for i_neg_thresh in neg_thresh:
        for i_pos_thresh in pos_thresh:
            timing_analyzer.sort_data(i_neg_thresh, i_pos_thresh)
            for i_sell_adj in sell_adj:
                for i_buy_adj in buy_adj:
                    timing_analyzer.do_stat(i_sell_adj, i_buy_adj)
                    metric = timing_analyzer.do_test()
                    if (metric > best_score):
                        best_score = metric
                        best_neg_thresh = i_neg_thresh
                        best_pos_thresh = i_pos_thresh
                        best_sell_adj = i_sell_adj
                        best_buy_adj = i_buy_adj
                
    timing_analyzer.sort_data(best_neg_thresh, best_pos_thresh)
    stats, sell_trigger, buy_trigger = timing_analyzer.do_stat(best_sell_adj, best_buy_adj)
    metric = timing_analyzer.do_test()
    p_hist, p = timing_analyzer.plot_data(doHTML)
    
    results_df = pd.DataFrame()
    results_df['Sell Trigger'] = pd.Series(sell_trigger)
    results_df['Buy Trigger'] = pd.Series(buy_trigger)
    results_df['Metric'] = pd.Series(metric)
    results_df = results_df.round(3)
    
    return stats, results_df, p_hist, p, best_neg_thresh, best_pos_thresh, \
        best_sell_adj, best_buy_adj