# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:36:12 2022

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

############ Extraction and Loading ######################

# plt.plot(x, output[' festival'])
# plt.plot(x, linregress(x, output[' festival'])[0]*x+linregress(x, output[' festival'])[1])
# plt.plot(x, output_adj[' festival'])
# plt.plot(x[155:], Zscores[' festival'])
# plt.show()

""""
Here we are going to loop through each basket of keywords, run the model and 
save an excel with the data that is going to be used later for the plotting
"""
for l in range (9,22): #len(data_input.index)
    name = data_input.iloc[l,0]#Name of the theme
    country = data_input.iloc[l,1]# specify full name of the country
    keywords = data_input.iloc[l,2].split(",") # specify name of the list of keywords
    print(name, country, keywords)

    # retrieve historical data for list of keywords of choice
    output=extract_GT(keywords,country)
    x = np.arange(0,output.shape[0])
    output['x'] = 0
    # Get week number
    output['week'] = output.index.map(lambda x: x.isocalendar()[1])
    
    #Initial value per week = Mean over the last 7 years
    output_init = output[:'2020-01-01'].groupby('week').mean()
    
    ############################## Model #####################################
    # Comparison of the data
    """
    In this section we will compare the value from 2020/01/01 to today (Covid Period)
    witht the value before the covid appeared
    We will then compute the diff in basis point
    """
    print("... Comparison")
    # Create df witht the diff in Bp
    output_adj = pd.DataFrame().reindex_like(output)

    
    # Adjust data for each keyword
    for col in tqdm(keywords):
        for t in output['2020-01-01':].index:
            week = output.loc[t,'week']
            if week != 53:
                data = (output.loc[t,col]-output_init.loc[week,col])/output.std()[col]
                output_adj.loc[t,col] = data
            else:
                    output_adj.loc[t,col] =data 
    output_adj["2020-01-01":].to_excel("Z_scores GT\data_temp_"+name+".xlsx")


