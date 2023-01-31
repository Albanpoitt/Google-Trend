# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 09:07:42 2022

@author: banba
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress
import datetime
from openpyxl import load_workbook
from statsmodels.graphics.tsaplots import plot_acf

from pytrends.request import TrendReq
pytrend = TrendReq()

############## Functions ##################

def extract_GT(list_kw,country_name):
    """
    define function to get last 5 years weekly data for a list of keywords
    """
    # load weekly historical data for the last 5 years, i.e., 260 weeks
    pytrend.build_payload(kw_list = list_kw[:1], geo=country_dic[country_name],timeframe='today 5-y')
    # load historical data for the first keyword
    data = pytrend.interest_over_time().drop(columns=["isPartial"])
    # make sure we only have 260 entries
    data = data.iloc[:260]
    
    if len(list_kw) == 1:
        return data
    else:
        for kw in tqdm(list_kw[1:]):
            pytrend.build_payload(kw_list = [kw], geo=country_dic[country_name],timeframe='today 5-y')
            temp = pytrend.interest_over_time().drop(columns=["isPartial"]).iloc[:260]
            data = data.merge(temp,how="left",left_index=True,right_index=True)
        return data


def zscore(arr,window):
    """
    Define function to compute Z-score of a given Series
    Calculate Z-scores for rolling windows of N periods (weeks)
    """
    rol = arr.rolling(window=window, min_periods=window)
    avg = rol.mean() #.shift(1) to not include same period
    std = rol.std(ddof=0) #.shift(1) to not include same period
    zscore = (arr - avg) / std
    return zscore
# create empty DF in which to store the Z-scores


################## INPUTS ##################################
#Read excel with all the inputs
data_input = pd.read_excel("list_input.xlsx", header=0)
# Country codes: To be updated if more countries are used
country_dic = {'United States':'US','United Kingdom':'GB','Germany':'DE','France':'FR','Spain':'ES'}

# ############ Extraction and Loading ######################

plt.plot(x, output[' festival'], label='init')
plt.plot(x[25:], moving_avg[' festival'], label= 'mv')
plt.plot(x, params[' festival'][1]*x+params[' festival'][0])
plt.plot(x, output_adj[' festival'], label='adj')
# plt.plot(x[155:], Zscores[' festival'])
plt.legend()
plt.show()

""""
Here we are going to loop through each basket of keywords, run the model and 
save an excel with the data that is going to be used later for the plotting
"""
for l in range (7,8): #len(data_input.index)
    name = data_input.iloc[l,0]#Name of the theme
    country = data_input.iloc[l,1]# specify full name of the country
    keywords = data_input.iloc[l,2].split(",") # specify name of the list of keywords
    print(name, country, keywords)

    # retrieve historical data for list of keywords of choice
    output=extract_GT(keywords,country)
    
    ############################## Model #####################################
    # Define regression independent variable
    x = np.arange(0,output.shape[0])
    output['x'] = 0
    
    #Moving Avg Data for the trend 
    moving_avg = pd.DataFrame().reindex_like(output)
    moving_avg = output.rolling(window=26, min_periods=26).mean().dropna()
    
    # Calculate and store intercept and slope for each keyword
    print("... Calculating baselines")
    params = {}
    for kw in keywords:
        slope, intercept, r, p, se = linregress(x[25:], moving_avg[kw])
        params[kw] = (intercept,slope)
    
    # Get week number
    output['week'] = output.index.map(lambda x: x.isocalendar()[1])
    
    # SEASONALLY ADJUST DATA
    """
    To take into account the seasonality of the data and deseasonalised it
    We used here the X11, more part of it and the idea behind it 
    http://www.christophsax.com/x13story/x11.pdf
    
    The code find in the model part of the code used this method X11
    """
    print("... Seasonally adjusting")
    # Adjust data on weekly seasonality
    output_adj = pd.DataFrame().reindex_like(output)
    output_adj['week'] = output['week']
    output_adj['x'] = output['x']
    
    # Adjust data for each keyword
    for col in tqdm(keywords):
        t0 = 1
        t = 0
        i = 1
        while i<6:
            while t < 52*i:
                cycle_avg = output[col][52*(i-1):52*i].mean()
                output_adj[col][t] = cycle_avg + params[col][1]*((t+1) - t0 - (52-1)/2)
                t = t + 1
            i = i + 1
            t0 = t + 1

    # Z SCORES
    """
    Rolling window of 4 months or 16 weeks to reduce noise
    """
    print("... Calculating Z-scores")
    Zscores = pd.DataFrame(index=output_adj.index, columns=keywords)
    # Length of period (weeks)
    N = 16 # --> INPUT
    for col in Zscores.columns:
        Zscores[col] = zscore(output_adj[col],N)
    # Filter data since beginning 2020
    Zscores = Zscores['2020':]
    #Save
    Zscores.to_excel("Z_scores GT\data_temp_"+name+".xlsx")
