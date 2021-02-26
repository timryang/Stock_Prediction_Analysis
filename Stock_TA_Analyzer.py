# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 18:52:27 2021

@author: timot
"""


from CommonFunctions.commonFunctions import *
import ta
from scipy.signal import savgol_filter
from sklearn import svm

#%% Inputs

class Stock_TA_Analyzer():
    
    def __init__(self):
        self.svr_clf_ = svm.SVR()
        self.stock_data_ = pd.DataFrame()
        self.dates_ = np.array([])
        self.close_data_ = np.array([])
        self.close_smooth_ = np.array([])
        self.obv_slope_ = np.array([])
        self.obv_slope_smooth_ = np.array([])
        
    def collect_data(self,ticker,start_date):
        today = datetime.date.today()
        self.stock_data_ = collect_stock_data(ticker, start_date, today.strftime('%m-%d-%Y'))
        self.dates_ = np.array(pd.to_datetime(self.stock_data_['Date']))
        self.close_data_ = np.array(self.stock_data_['Close'])
        
    def load_data(self,stock_data):
        self.stock_data_ = stock_data
        
    def compute_ta(self,slow_ema=26,fast_ema=12,signal_ema=9,rsi_sell_thresh=70,rsi_buy_thresh=50,\
                   filter_win_length=5,filter_polyorder=1,isBokeh=False):
        
        self.macd_obj_ = ta.trend.MACD(self.stock_data_['Close'],slow_ema,fast_ema,signal_ema)
        self.rsi_obj_ = ta.momentum.rsi(self.stock_data_['Close'])
        obv_obj = ta.volume.on_balance_volume(self.stock_data_['Close'],self.stock_data_['Volume'])
        
        self.close_smooth_ = savgol_filter(self.close_data_,filter_win_length,filter_polyorder)
        
        macd_buy = self.dates_[np.where(np.sign(np.array(self.macd_obj_.macd_diff()[:-1]))<np.sign(np.array(self.macd_obj_.macd_diff()[1:])))[0]]
        macd_sell = self.dates_[np.where(np.sign(np.array(self.macd_obj_.macd_diff()[:-1]))>np.sign(np.array(self.macd_obj_.macd_diff()[1:])))[0]]
        macd_buy = list(np.tile(macd_buy,(2,1)).T)
        macd_sell = list(np.tile(macd_sell,(2,1)).T)
        
        rsi_buy = self.dates_[np.where(self.rsi_obj_<rsi_buy_thresh)[0]]
        rsi_sell = self.dates_[np.where(self.rsi_obj_>rsi_sell_thresh)[0]]
        rsi_sell_line = rsi_sell_thresh*np.ones(2)
        rsi_buy_line = rsi_buy_thresh*np.ones(2)
        
        obv_smooth = savgol_filter(obv_obj,filter_win_length,filter_polyorder)
        obv_slope = np.diff(obv_smooth)
        self.obv_slope_ /= max(obv_slope)
        self.obv_slope_smooth_ = savgol_filter(obv_slope,filter_win_length,filter_polyorder)
        
        vert_line = np.array([np.min(self.close_data_),np.max(self.close_data_)])
        x_date_line = [self.dates_[0],self.dates_[-1]]
        vert_point = np.min(self.close_data_)-0.01*np.min(self.close_data_)
        
        macd_p = plot_values([self.dates_,self.dates_], [self.macd_obj_.macd(),self.macd_obj_.macd_signal()],\
                             ['MACD','Signal'], 'Date', 'MACD', 'MACD', True, True, ['-'], isBokeh)
        rsi_p = plot_values([self.dates_,x_date_line,x_date_line], [self.rsi_obj_,rsi_sell_line,rsi_buy_line],\
                            ['RSI','Sell Thresh','Buy Thresh'], 'Date', 'RSI', 'RSI', True, True, ['-'], isBokeh)
        obv_p = plot_values([self.dates_,self.dates_], [obv_obj,obv_smooth], ['OBV Raw','OBV Smooth'],\
                            'Date', 'OBV', 'OBV', True, True, ['-'], isBokeh)
        
        plt1_x = [self.dates_,self.dates_]+macd_buy+macd_sell+[rsi_buy,rsi_sell]
        plt1_y = [self.close_data_,self.close_smooth_]+[vert_line]*(len(macd_buy)+len(macd_sell))\
            +[np.tile(vert_point,len(rsi_buy)),np.tile(vert_point,len(rsi_sell))]
        plt1_labels = ['Close Raw','Close Smooth']+['MACD Buy']*len(macd_buy)+['MACD Sell']*len(macd_sell)\
            +['RSI Buy','RSI Sell']
        if isBokeh:
            plt1_lines = ['0solid','1solid']+['2dashed']*len(macd_buy)+['3dashed']*len(macd_sell)+['2dot','3dot']
        else:
            plt1_lines = ['C0-','C1-']+['g--']*len(macd_buy)+['r--']*len(macd_sell)+['go','ro']
        analysis_p = plot_values(plt1_x, plt1_y, plt1_labels, 'Date', 'Close', 'Analysis', True, False, plt1_lines,isBokeh)
        
        return macd_p,rsi_p,obv_p,analysis_p
    
    def preprocess_and_train(self,days_ahead=4,days_evaluate=3,train_size=0.5,do_smooth=True,\
                             kernel='rbf',C=0.1,eps=1e-5,degree=3,isBokeh=False):
        
        # Form data
        close_diff_perc_smooth = (self.close_smooth_[days_ahead:]-self.close_smooth_[:-days_ahead])/self.close_smooth_[:-days_ahead] #percent change
        close_diff_perc = (self.close_data_[days_ahead:]-self.close_data_[:-days_ahead])/self.close_data_[:-days_ahead] #percent change
        macd_diff = self.macd_obj_.macd_diff()
        
        # Standardize
        macd_diff_std = (macd_diff-np.mean(macd_diff))/np.std(macd_diff)
        rsi_std = (self.rsi_obj_-np.mean(self.rsi_obj_))/np.std(self.rsi_obj_)
        if do_smooth:
            obv_slope_std =  (self.obv_slope_smooth_-np.mean(self.obv_slope_smooth_))/np.std(self.obv_slope_smooth_)
        else:
            obv_slope_std =  (self.obv_slope_-np.mean(self.obv_slope_))/np.std(self.obv_slope_)
        
        # Create input/output matrix
        first_idx = np.max([macd_diff.first_valid_index(),self.rsi_obj_.first_valid_index()]) #first valid idx
        m_size = macd_diff.size-(first_idx+days_evaluate+days_ahead-1) #rows (days)
        n_size = days_evaluate*3 #columns (n_features)
        if do_smooth:
            output = close_diff_perc_smooth[first_idx+days_evaluate-1:]
        else:
            output = close_diff_perc[first_idx+days_evaluate-1:]
        input_array = np.zeros((m_size,n_size))
        for i_m in range(m_size):
            start_idx = first_idx+i_m
            input_array[i_m,:] = np.r_[macd_diff_std[start_idx:start_idx+days_evaluate],\
                                       rsi_std[start_idx:start_idx+days_evaluate],\
                                           obv_slope_std[start_idx-1:start_idx-1+days_evaluate]]
        
        x_train = input_array[:int(m_size*train_size),:]
        y_train = output[:int(m_size*train_size)]
        x_test = input_array[int(m_size*train_size):,:]
        y_test = output[int(m_size*train_size):]
        
        predict_input = np.zeros((days_ahead,n_size))
        for i_m in range(days_ahead):
            start_idx = macd_diff_std.size-(days_ahead+days_evaluate-1)+i_m
            predict_input[i_m,:] = np.r_[macd_diff_std[start_idx:start_idx+days_evaluate],\
                                          rsi_std[start_idx:start_idx+days_evaluate],\
                                              obv_slope_std[start_idx-1:start_idx-1+days_evaluate]]
                
        self.svr_clf_ = svm.SVR(kernel=kernel,degree=degree,C=C,epsilon=eps)
        self.svr_clf_.fit(x_train,y_train)
        train_output = self.svr_clf_.predict(x_train)
        test_output = self.svr_clf_.predict(x_test)
        predict_output = self.svr_clf_.predict(predict_input)
        
        predict_dates = [pd.to_datetime(self.dates_[-1])]
        for i_day in range(1,days_ahead+1):
            predict_dates.append(next_business_day(predict_dates[-1]))
        predict_dates = predict_dates[1:]
        
        actual = close_diff_perc[first_idx+days_evaluate-1:]
        dates_diff = self.dates_[first_idx+days_evaluate+days_ahead-1:]
        dates_train = dates_diff[:int(m_size*train_size)]
        dates_test = dates_diff[int(m_size*train_size):]
        x_values = [dates_diff,dates_diff,dates_train,dates_test,predict_dates]
        y_values = [actual,output,train_output,test_output,predict_output]
        labels = ['Actual','Output','Train','Test','Predict']
        if isBokeh:
            plt_lines = ['0dashed','1solid','3solid','3dashed','4solid']
        else:
            plt_lines = ['C0--','C1-','C3-','C3--','C4-']
        p_svr = plot_values(x_values, y_values, labels, 'Date', 'Percent Change', 'SVR Results',\
                    isDates=True, default_lines=False, lines=plt_lines, isBokeh=isBokeh)
            
        return p_svr