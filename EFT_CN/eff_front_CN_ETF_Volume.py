# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:33:18 2022

@author: timyu
"""

import numpy as np
import pandas as pd
import scipy.optimize as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

import datetime as dt
from pandas.tseries.offsets import BDay

from pandas_datareader import data as pdr

def getData_Volume(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    volumeData = stockData['Volume']
    meanVolume = volumeData.mean(skipna=True)
    
    # filter funds with Volume >= 10,000,000
    meanVolume = meanVolume[meanVolume >= 1e7]
    
    return meanVolume.index.tolist()

# 252 trading days per year 
days = 30
endDate = dt.datetime.today() - BDay(1) # start date: previous bussiness day
startDate = endDate - dt.timedelta(days=days)

fund_SS = pd.read_csv("fundList_SS.csv", header=None)
fund_SZ = pd.read_csv("fundList_SZ.csv", header=None)

fundList_SS = [str(fund) for fund in fund_SS.iloc[:, 0]]
fundList_SZ = [str(fund) for fund in fund_SZ.iloc[:, 0]]

stocks_SS = [stock+'.SS' for stock in fundList_SS]
stocks_SZ = [stock+'.SZ' for stock in fundList_SZ]

stocks_all = stocks_SS + stocks_SZ

'''
Remove funds with no data on yahoo finance
'''
rm_list = pd.read_csv('list_nan.txt', sep=' ', header=None, )
rm_list = rm_list.drop(range(1, 168, 2))
stock_rm = [str(fund)[1:10] for fund in rm_list.iloc[:, 6]]


# remove NaN stocks
stocks = [stock for stock in stocks_all if stock not in stock_rm]

print("Load Data...")
pd.DataFrame(getData_Volume(stocks=stocks, start=startDate, end=endDate)).to_csv("fundVolumeFilter.csv", header=False, index=False)
print("Data Loaded.")