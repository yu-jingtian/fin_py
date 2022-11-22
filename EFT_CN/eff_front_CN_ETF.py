# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 01:41:26 2022

@author: timyu
"""

import numpy as np
import datetime as dt
import pandas as pd
import scipy.optimize as sc
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

from pandas_datareader import data as pdr

def getData(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
    stockData = stockData['Close']
    
    keep_list = []
    for i in range(stockData.shape[1]):
        # keep the cols with missing <= 2
        keep_list.append(sum(stockData.iloc[:,i].isna()) <= 2)    
    stockData = stockData.iloc[:, keep_list]
    
    # mean imputation
    stockData = stockData.fillna(stockData.mean())
    
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix


def portfolioPerformance(weights, meanReturns, covMatrix):
    returns = np.sum(meanReturns*weights)*252
    std = np.sqrt(np.dot(weights.T,np.dot(covMatrix, weights)))*np.sqrt(252)
    return returns, std


def negativeSR(weights, meanReturns, covMatrix, riskFreeRate = 0):
    pReturns, pStd = portfolioPerformance(weights, meanReturns, covMatrix)
    return - (pReturns - riskFreeRate)/pStd

def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintSet=(0,1)):
    "Minimize the negative SR, by altering the weights of the portfolio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(negativeSR, numAssets*[1./numAssets], args=args,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVariance(weights, meanReturns, covMatrix):
    return portfolioPerformance(weights, meanReturns, covMatrix)[1]
def minimizeVariance(meanReturns, covMatrix, constraintSet=(0,1)):
    """Minimize the portfolio variance by altering the 
     weights/allocation of assets in the portfolio"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioReturn(weights, meanReturns, covMatrix):
        return portfolioPerformance(weights, meanReturns, covMatrix)[0]

def efficientOpt(meanReturns, covMatrix, returnTarget, constraintSet=(0,1)):
    """For each returnTarget, we want to optimise the portfolio for min variance"""
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type':'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintSet
    bounds = tuple(bound for asset in range(numAssets))
    effOpt = sc.minimize(portfolioVariance, numAssets*[1./numAssets], args=args, method = 'SLSQP', bounds=bounds, constraints=constraints)
    return effOpt


def calculatedResults(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
        
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    return maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns


def EF_graph(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns = calculatedResults(meanReturns, covMatrix, riskFreeRate, constraintSet)
    #Max SR
    MaxSharpeRatio = go.Scatter(
        name='Maximium Sharpe Ratio',
        mode='markers',
        x=[maxSR_std],
        y=[maxSR_returns],
        marker=dict(color='red',size=14,line=dict(width=3, color='black'))
    )
    #Min Vol
    MinVol = go.Scatter(
        name='Mininium Volatility',
        mode='markers',
        x=[minVol_std],
        y=[minVol_returns],
        marker=dict(color='green',size=14,line=dict(width=3, color='black'))
    )
    #Efficient Frontier
    EF_curve = go.Scatter(
        name='Efficient Frontier',
        mode='lines',
        x=[round(ef_std*100, 2) for ef_std in efficientList],
        y=[round(target*100, 2) for target in targetReturns],
        line=dict(color='black', width=4, dash='dashdot')
    )
    
    data = [MaxSharpeRatio, MinVol, EF_curve]
    layout = go.Layout(
        title = 'Portfolio Optimisation with the Efficient Frontier',
        yaxis = dict(title='Annualised Return (%)'),
        xaxis = dict(title='Annualised Volatility (%)'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder='normal',
            bgcolor='#E2E2E2',
            bordercolor='black',
            borderwidth=2),
        width=800,
        height=600)
    
    fig = go.Figure(data=data, layout=layout)
    return fig.show()




def rsltFun(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
    for target in targetReturns:
        efficientList.append(efficientOpt(meanReturns, covMatrix, target)['fun'])
        
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    rslt = [maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns, riskFreeRate]
    return rslt


def rsltFun_no_EF(meanReturns, covMatrix, riskFreeRate=0, constraintSet=(0,1)):
    """Read in mean, cov matrix, and other financial information
        Output, Max SR , Min Volatility, efficient frontier """
    # Max Sharpe Ratio Portfolio
    maxSR_Portfolio = maxSR(meanReturns, covMatrix)
    maxSR_returns, maxSR_std = portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,0) for i in maxSR_allocation.allocation]
    
    # Min Volatility Portfolio
    minVol_Portfolio = minimizeVariance(meanReturns, covMatrix)
    minVol_returns, minVol_std = portfolioPerformance(minVol_Portfolio['x'], meanReturns, covMatrix)
    minVol_allocation = pd.DataFrame(minVol_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVol_allocation.allocation = [round(i*100,0) for i in minVol_allocation.allocation]
    
    # Efficient Frontier
    efficientList = []
    targetReturns = np.linspace(minVol_returns, maxSR_returns, 20)
        
    maxSR_returns, maxSR_std = round(maxSR_returns*100,2), round(maxSR_std*100,2)
    minVol_returns, minVol_std = round(minVol_returns*100,2), round(minVol_std*100,2)
    rslt = [maxSR_returns, maxSR_std, maxSR_allocation, minVol_returns, minVol_std, minVol_allocation, efficientList, targetReturns, riskFreeRate]
    return rslt


def rsltOutput(rslt, startDate, endDate):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    maxSR_returns = rslt[0] 
    maxSR_std = rslt[1]
    maxSR_allocation = rslt[2]
    minVol_returns = rslt[3]
    minVol_std = rslt[4]
    minVol_allocation = rslt[5]
    riskFreeRate = rslt[8]
    
    
    maxSR_allocation = maxSR_allocation.T
    minVol_allocation = minVol_allocation.T
    
    allocation = pd.concat([maxSR_allocation, minVol_allocation])
    allocation.insert(0, "Portfolio Composition (%)", ["Max SR", "Min Vol"], True)
    
    info = pd.DataFrame({'Start Date': [startDate.strftime('%Y-%m-%d')],
                         'End Date': [endDate.strftime('%Y-%m-%d')],
                         'Risk Free Rate': [riskFreeRate]})
    
    returnVol = pd.DataFrame({'Portfolio': ["Max SR", "Min Vol"],
                              'Sharpe Ratio': [maxSR_returns/maxSR_std, minVol_returns/minVol_std],
                              'Returns (%)': [maxSR_returns, minVol_returns],
                              'Volatility (%)': [maxSR_std, minVol_std],})
    
    return(info, returnVol, allocation)

def allocSumFun(allocation):
    alloc_SR = allocation.iloc[[0]]
    alloc_Vol = allocation.iloc[[1]]
    
    alloc_SR = alloc_SR.loc[:, (alloc_SR != 0).any(axis=0)]
    alloc_Vol = alloc_Vol.loc[:, (alloc_Vol != 0).any(axis=0)]
    return(alloc_SR, alloc_Vol)

def EFPlot(meanReturns, covMatrix, rslt, startDate, endDate, country = "US"):
    """Return a graph ploting the min vol, max sr and efficient frontier"""
    maxSR_returns = rslt[0] 
    maxSR_std = rslt[1]
    minVol_returns = rslt[3]
    minVol_std = rslt[4]
    efficientList = rslt[6]
    targetReturns = rslt[7]
    riskFreeRate = rslt[8]
    
    stocks = list(meanReturns.index)
    
    returns = round(meanReturns*252*100, 2)
    std = np.round(np.sqrt(np.diag(covMatrix.values)*252)*100, 2)
    stockDf = pd.DataFrame({'Name': stocks,
                            'Returns': list(returns),
                            'Vol': list(std)})
    
    
    returnVol = pd.DataFrame({'Name': ["Max SR", "Min Vol"],
                              'Returns': [maxSR_returns, minVol_returns],
                              'Vol': [maxSR_std, minVol_std],})
    '''
    df_EF = pd.DataFrame({'Returns': [round(target*100, 2) for target in targetReturns],
                          'Vol': [round(ef_std*100, 2) for ef_std in efficientList]})
    '''
    
    plt.figure(figsize=(10,8))
    plt.title("Country: {}, Start Date: {}, End Date: {}, Risk Free Rate: {}".format(country,
                                                                                     startDate.strftime('%Y-%m-%d'),
                                                                                     endDate.strftime('%Y-%m-%d'),
                                                                                     riskFreeRate))
    #plt.plot(df_EF['Vol'], df_EF['Returns'], 'k--', label="Efficient Frontier")
    plt.plot(returnVol['Vol'][0], returnVol['Returns'][0], 'go', label="Max Sharpe Ratio")
    plt.plot(returnVol['Vol'][1], returnVol['Returns'][1], 'co', label="Min Volatility")
    '''
    for i in range(len(returnVol)):
        plt.text(returnVol['Vol'][i], returnVol['Returns'][i], returnVol['Name'][i],
                 #bbox=dict(facecolor='blue', alpha=0.3),
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize='small')
    '''
    plt.plot(stockDf['Vol'], stockDf['Returns'], 'bo')
    
    '''
    for i in range(len(stockDf)):
        plt.text(stockDf['Vol'][i], stockDf['Returns'][i], stockDf['Name'][i],
                 #bbox=dict(facecolor='blue', alpha=0.3),
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 fontsize='medium')
        
    '''
    plt.legend()
    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    tt = round(time.time())/1000
    plt.savefig('EF_plot{}.jpg'.format(tt), dpi=200)
    
    return plt.show()

# 252 trading days per year 
days = 30
endDate = dt.datetime.today() - dt.timedelta(days=1) # start date: yesterday
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
meanReturns, covMatrix = getData(stocks=stocks, start=startDate, end=endDate)
print("Data Loaded.")


print("Optimization start...")
rslt = rsltFun_no_EF(meanReturns, covMatrix, riskFreeRate=0.03, constraintSet=(0,1))
print("Optimization done.")

print("Result Presentation...")
info, returnVol, allocation = rsltOutput(rslt, startDate, endDate)
alloc_SR, alloc_Vol = allocSumFun(allocation)

print(info.to_markdown(index=False))
print(returnVol.to_markdown(index=False))
print(alloc_SR.to_markdown(index=False))
print(alloc_Vol.to_markdown(index=False))


print("Plotting...")
EFPlot(meanReturns, covMatrix, rslt, startDate, endDate, country="CN")
print("Done.")

