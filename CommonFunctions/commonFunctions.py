# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:03:33 2020

@author: timot
"""
# import GetOldTweets3 as got
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from operator import itemgetter
import warnings
from nltk.corpus import stopwords
import time
import datetime
from bokeh.plotting import figure
from bokeh.palettes import Category10_10 as palette
import bokeh.layouts
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from sklearn.model_selection import cross_validate
from scipy import stats
import holidays
from bs4 import BeautifulSoup
import requests
import snscrape.modules.twitter as sntwitter

plt.rcParams.update({'font.size': 22})

warnings.filterwarnings("ignore")

#%% Common functions to import

#%% Common functions

def prev_business_day(day_ref):
    prev_day = day_ref-datetime.timedelta(days=1)
    while prev_day.weekday() in holidays.WEEKEND or prev_day in holidays.US():
        prev_day -=  datetime.timedelta(days=1)
    return prev_day

def next_business_day(day_ref):
    next_day = day_ref + datetime.timedelta(days=1)
    while next_day.weekday() in holidays.WEEKEND or next_day in holidays.US():
        next_day += datetime.timedelta(days=1)
    return next_day

def rename_date_field(df):
    dfColumns = df.columns
    checkDateField = ['date' in tempStr.lower() for tempStr in dfColumns]
    df.rename(columns = {dfColumns[checkDateField][0]: 'Date'}, inplace = True)

def find_idx(input_list, condition):
    return [idx for idx, val in enumerate(input_list) if val == condition]

def del_idx(input_list, idxs):
    for idx in sorted(idxs, reverse=True):
        del input_list[idx]
    return input_list

def parse_input(txt_input, blank_fill, expect_output='float'):
    if txt_input == '' or txt_input == None:
        output = [blank_fill]
    elif ':' in txt_input:
        if expect_output == 'float':
            range_vals = [float(val) for val in txt_input.split(':')]
        elif expect_output == 'int':
            range_vals = [int(val) for val in txt_input.split(':')]
        output = list(np.arange(range_vals[0], range_vals[2]+range_vals[1], range_vals[1]))
    else:
        if expect_output == 'float':
            output = [float(val) for val in txt_input.split(',')]
        elif expect_output == 'int':
            output = [int(val) for val in txt_input.split(',')]
    return output
    

#%% Twitter functions

def get_tweets_list(txtSearch, startDate=None, stopDate=None, maxTweets=None, userName=None, lang='en'):
    
    # Creating list to append tweet data to
    tweetsList = []
    
    if startDate == None:
        startDate = datetime.datetime.now()
    else:
        startDate = datetime.datetime.strptime(startDate, '%m-%d-%Y')
    if stopDate == None:
        stopDate = datetime.datetime.now()
    else:
        stopDate = datetime.datetime.strptime(stopDate, '%m-%d-%Y')
    
    dateRange = [stopDate-datetime.timedelta(days=x) for x in range((stopDate-startDate).days+1)]
    
    for iDate in dateRange:
        # Using TwitterSearchScraper to scrape data and append tweets to list
        tempStart = iDate.strftime('%Y-%m-%d')
        tempStop = (iDate+datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        searchString = txtSearch + ' since:' + tempStart + ' until:' + tempStop + ' lang:' + lang
        if userName!=None:
            searchString = searchString + ' from:' + userName
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(searchString).get_items()):
            if maxTweets!=None and i>maxTweets:
                break
            tweetsList.append([tweet.date, tweet.id, tweet.content, tweet.user.username,\
                               tweet.retweetCount, tweet.likeCount])
        
    return tweetsList

def get_tweets(txtSearch, startDate=None, stopDate=None, maxTweets=None, userName=None, lang='en'):
    
    tweetsList = get_tweets_list(txtSearch, startDate, stopDate, maxTweets, userName, lang)
    
    # Creating a dataframe from the tweets list above 
    tweetsDF = pd.DataFrame(tweetsList, columns=['Date', 'ID', 'Text', 'Username', 'Retweets', 'Likes'])
    
    tweetsDF.sort_values(by = ['Date'], inplace = True)
    tweetsDF.reset_index(drop = True, inplace = True)
    return tweetsDF

def count_tweets_by_day(tweets_DF):
    tweet_dates = [str(dt.date()) for dt in pd.to_datetime(tweets_DF['Date'])]
    unique_dates = list(set(tweet_dates))
    unique_dates.sort()
    num_daily_tweets = [len(find_idx(tweet_dates, date)) for date in unique_dates]
    return unique_dates, num_daily_tweets

def predict_from_tweets(clf, count_vect, tfTransformer, txt_search,\
                        userName=None, num_max_tweets=0, printAll=False):
    predictTweets = get_tweets(txt_search, maxTweets=num_max_tweets)
    tweetText = list(predictTweets['Text'])
    tf_text = transform_text(tweetText, count_vect, tfTransformer)
    predictions = clf.predict(tf_text)
    predictionTxt = ""
    if printAll:
        for idx, prediction in enumerate(predictions):
            predictionTxt = predictionTxt + "Tweet: " + "\n" + tweetText[idx] + \
                "\n" + "Prediction: " + prediction + "\n\n"
        print("\n" + predictionTxt)
    classes = clf.classes_
    class_counts = []
    for i_class in classes:
        i_num = list(predictions).count(i_class)
        class_counts.append(i_num)
    result = "Predicted Tweets: "
    for idx, i_class in enumerate(classes):
        result = result + "\n" + i_class + ": " + str(class_counts[idx])
    print("\n" + result)
    return result, predictionTxt

#%% Stock functions

def collect_stock_data(ticker, start_date, end_date=None):
        start_unix = datetime.datetime.timestamp(datetime.datetime.strptime(start_date, '%m-%d-%Y'))
        if end_date==None:
            end_unix = time.time()
        else:
            end_unix = datetime.datetime.timestamp(datetime.datetime.strptime(end_date, '%m-%d-%Y'))
        tempLink = "https://query1.finance.yahoo.com/v7/finance/download/" + ticker\
            + "?period1=" + str(int(start_unix)) + "&period2="\
                + str(int(end_unix)) + "&interval=1d&events=history"
        try:
            stockData = pd.read_csv(tempLink)
        except:
            raise ValueError("Bad link: Double check your ticker")
        return stockData
    
#%% Plot functions

def plot_values(x_values, y_values, labels, x_label, y_label, title, isDates, default_lines=True, lines=['-'], isBokeh=False):
    if default_lines:
        lines = np.tile(np.array(lines),len(labels))
    if isBokeh:
        if isDates:
            axis_type = "datetime"
        else:
            axis_type = "linear"
        p = figure(tools="pan,box_zoom,reset,save", title=title,\
           x_axis_label=x_label, y_axis_label=y_label, x_axis_type=axis_type,\
               width=525, height=310)
        for idx, i_y_values in enumerate(y_values):
            if lines[idx] == '-':
                p.line(x_values[idx], i_y_values, legend_label=labels[idx], color=palette[idx])
            else:
                color = palette[int(lines[idx][0])]
                style = lines[idx][1:]
                if style=='dot':
                    p.circle(x_values[idx],i_y_values, legend_label=labels[idx], color=color, size=5)
                else:    
                    p.line(x_values[idx], i_y_values, legend_label=labels[idx], color=color, line_dash=style)
        p.legend.location = "top_left"
        return p
    else:      
        plt.figure(figsize=(20,10))
        if isDates:
            formatter = mdates.DateFormatter("%m-%d-%Y")
            locator = mdates.DayLocator(bymonthday=[1, 15])
            ax = plt.gca()
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(locator)
            for idx, i_y_values in enumerate(y_values):
                ax.plot_date(x_values[idx], i_y_values, lines[idx], label=labels[idx])
            plt.xticks(rotation = 70)
        else:
            for idx, i_y_values in enumerate(y_values):
                ax.plot(x_values[idx], i_y_values, lines[idx], label=labels[idx])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.grid()
        handles, labels_return = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_return, handles))
        ax.legend(by_label.values(), by_label.keys())
        plt.show()
        return "used matplotlib"

def plot_hist(values, bin_interval, x_label, y_label, title, isBokeh):
    if isBokeh:
        bins = int(np.ceil(np.max(values)/bin_interval))
        hist, edges = np.histogram(values, density=True, bins=bins)
        p = figure(tools="pan,box_zoom,reset,save", title=title,\
           x_axis_label=x_label, y_axis_label='Percentage',\
               width=350, height=310)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
        return p
    else:
        bins = np.arange(np.min(values), np.max(values), bin_interval)
        plt.hist(values, bins = bins)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
        return "used matplotlib"
        
def plot_multi_scatter(axes_values_list, labels, axes_labels, title, color_values_list=None, saveFig=False):
    # Input axes_values as a list of arrays with n dimensions
    fig = plt.figure()
    colors = ['r','b','g','k','c','m','y'] # Only supports 7 labels
    markers = ['o','x','^'] # Only supports 3 labels
    cm = plt.cm.get_cmap('RdYlBu')
    if axes_values_list[0].ndim == 1:
        ax = fig.add_subplot(111)
        for idx, val in enumerate(labels):
            if hasattr(color_values_list[idx], "__len__"):
                p = ax.scatter(axes_values_list[idx], color_values_list[idx], marker=markers[idx], label=val)
            else:
                p = ax.scatter(axes_values_list[idx], np.zeros(axes_values_list[idx].shape), c='k', marker=markers[idx], label=val)
    elif axes_values_list[0].shape[1] == 2:
        ax = fig.add_subplot(111)
        for idx, val in enumerate(labels):
            if color_values_list == None:
                p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1], c=colors[idx], label=val)
            else:
                if hasattr(color_values_list[idx], "__len__"):
                    p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1],\
                               c=color_values_list[idx], marker=markers[idx], label=val, cmap=cm)
                    fig.colorbar(p)
                else:
                    p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1],\
                               c='k', marker=markers[idx], label=val, cmap=cm)
    elif axes_values_list[0].shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        for idx, val in enumerate(labels):
            if color_values_list == None:
                p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1], axes_values_list[idx][:,2],\
                            c=colors[idx], label=val)
            else:
                if hasattr(color_values_list[idx], "__len__"):
                    p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1], axes_values_list[idx][:,2],\
                                c=color_values_list[idx], marker=markers[idx], label=val, cmap=cm)
                    fig.colorbar(p)
                else:
                    p = ax.scatter(axes_values_list[idx][:,0], axes_values_list[idx][:,1], axes_values_list[idx][:,2],\
                                c='k', marker=markers[idx], label=val, cmap=cm)
        ax.set_zlabel(axes_labels[2])
    ax.set_xlabel(axes_labels[0])
    ax.set_ylabel(axes_labels[1])
    ax.set_title(title)
    ax.legend()
    plt.grid(b=True)
    if saveFig:
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        return '<img src="data:image/png;base64,{}">'.format(plot_url)
    else:
        plt.show()
        return "used matplotlib"
    
#%% Classifier functions

def downsample(x_values, y_values):
    total_list = list(zip(x_values, y_values))
    random.shuffle(total_list)
    x_rand, y_rand = zip(*total_list)
    x_rand = list(x_rand)
    y_rand = list(y_rand)
    y_unique = list(set(y_rand))
    num_ys = [y_rand.count(y) for y in y_unique]
    min_num = min(num_ys)
    for idx, y in enumerate(y_unique):
        num_del = num_ys[idx]-min_num
        y_idxs = find_idx(y_rand, y)
        remove_idxs = y_idxs[:num_del+1]
        y_rand = del_idx(y_rand, remove_idxs)
        x_rand = del_idx(x_rand, remove_idxs)
    return x_rand, y_rand

def transform_text(text, count_vect, tfTransformer):
    count_text = count_vect.transform(text)
    tf_text = tfTransformer.transform(count_text)
    return tf_text

def create_NB_text_classifier(x_total, y_total, trainSize, stopwordsList, useIDF,\
                              do_downsample=False, do_stat=True, n_features=10, doHTML=False):
    if do_downsample:
        x_total, y_total = downsample(x_total, y_total)
    x_train, x_test, y_train, y_test = train_test_split(
        x_total, y_total, train_size=trainSize, random_state=42)
    count_vect = CountVectorizer(stop_words = stopwordsList)
    x_train_counts = count_vect.fit_transform(x_train)
    tfTransformer = TfidfTransformer(use_idf=useIDF)
    x_train_tf = tfTransformer.fit_transform(x_train_counts)
    clf = MultinomialNB().fit(x_train_tf, y_train)
    
    if do_stat:
        x_test_tf = transform_text(x_test, count_vect, tfTransformer)
        report, p = classifier_statistics(x_test_tf, y_test, clf, doHTML=doHTML)
        most_inform = NB_show_most_informative(clf, count_vect, n_features=n_features, doHTML=doHTML)
    return clf, count_vect, tfTransformer, report, most_inform, p

#%% Classifier statistics

def classifier_statistics(x_test, y_truth, clf, doHTML=False):
    y_predict = clf.predict(x_test)
    class_names = clf.classes_
    report = classification_report(y_truth, y_predict, labels=class_names, output_dict=True)
    reportDF = pd.DataFrame()
    reportDF['Class/Metric'] = pd.Series(list(report.keys()))
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    total_support = report[list(report.keys())[3]]['support']
    for key, value in report.items():
        if (key != 'accuracy'):
            precision_list.append(round(value['precision'], 2))
            recall_list.append(round(value['recall'], 2))
            f1_list.append(round(value['f1-score'], 2))
            support_list.append(value['support'])
        else:
            precision_list.append('')
            recall_list.append('')
            f1_list.append(round(value, 2))
            support_list.append(total_support)
    reportDF['Precision'] = pd.Series(precision_list)
    reportDF['Recall'] = pd.Series(recall_list)
    reportDF['F1'] = pd.Series(f1_list)
    reportDF['Support'] = pd.Series(support_list)
    report = reportDF
    if not doHTML:                
        print("\nModel Results: ")
        print(report)
    p = show_confusion_matrix(y_truth, y_predict, labels=class_names, isBokeh=doHTML)
    return report, p

def show_confusion_matrix(y_truth, y_predict, labels, isBokeh=False):
    conf_matrix = confusion_matrix(y_truth, y_predict, labels=labels)
    if not isBokeh:
        sns.heatmap(conf_matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('Confusion Matrix')
        plt.show()
        return "matplotlib was used"
    else:
        cm_df = pd.DataFrame()
        cm_df['Predicted>'] = pd.Series(labels)
        for idx, label in enumerate(labels):
            cm_df[label] = pd.Series(list(conf_matrix[:, idx]))
        return cm_df

def NB_show_most_informative(clf, count_vect, n_features=10, doHTML=False):
    classes = clf.classes_
    num_classes = len(classes)
    features = count_vect.get_feature_names()
    probs = np.exp(clf.feature_log_prob_)
    top_features_list = []
    for idx in range(num_classes):
        complement_idx = np.where(np.array([range(num_classes)]) != idx)[1]
        complement_sum_probs = np.sum(probs[complement_idx], axis = 0)
        ratio = np.round(np.divide(probs[idx], complement_sum_probs), 3)
        top_features = (sorted(zip(features, ratio), key = itemgetter(1)))[:-(n_features+1):-1]
        top_features_list.append(top_features)
    if doHTML:
        report = pd.DataFrame()
        for idx_c, i_class in enumerate(classes):
            report[i_class] = pd.Series([str(feature) for feature in top_features_list[idx_c]])
    else:
        report = "";
        for i_class in classes:
            report = report + "{:<35s}".format(i_class)
        report = report + "\n"
        for idx_f in range(n_features):
            for idx_c in range(num_classes):
                report = report + "{:<35s}".format(str(top_features_list[idx_c][idx_f]))
            report = report + "\n"
        print("\nBelow printout gives the most informative words.")
        print("Example -> pos: ('gain', 3) indicates 'gain' is 3x more likely"\
              + " to predict pos compared to the sum prob of other classes.\n")
        print(report)
    return report

def do_cross_validate(clf, x_vectors, y_vector, metric='accuracy', k=5):
    total = np.hstack((x_vectors, y_vector.reshape(-1,1)))
    np.random.seed(seed=0)
    np.random.shuffle(total)
    
    x_vectors_shuff = total[:,:-1]
    y_vector_shuff = total[:,-1:]
    
    cv_results = cross_validate(clf, x_vectors_shuff, y_vector_shuff, scoring=metric, cv=k, return_estimator=True)
    
    test_scores = np.round(cv_results['test_score'], 2)
    test_scores_str = str(test_scores[0])
    for score in test_scores[2:]:
        test_scores_str = test_scores_str + '; ' + str(score)
    clf_out = cv_results['estimator'][np.argmax(test_scores)]
    
    return test_scores_str, clf_out

def extract_from_gridCV(clf, k):
    best_clf = clf.best_estimator_
    best_params = clf.best_params_
    results = pd.DataFrame.from_dict(clf.cv_results_)
    best_idx = clf.best_index_
    scores = [round(results['split'+str(i_k)+'_test_score'][best_idx],2) for i_k in range(k)]
    scores_str = str(scores[0])
    for score in scores[2:]:
        scores_str = scores_str + '; ' + str(score)
        
    return best_clf, best_params, scores_str