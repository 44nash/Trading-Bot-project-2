import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
from matplotlib import pyplot as plt



import pandas as pd
import numpy as np
import requests
from termcolor import colored as cl
from math import floor
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')



class MarcusIndicators(BaseEstimator, TransformerMixin):
    
    indicators = {'stochastic oscillator'}
    
    def __init__( self, window): 
        self.window = window

        
    def fit(self, X: pd.DataFrame, **fit_params):
        return self  # nothing to do!
    
    def transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        ticker = stochastic_oscillator(X)
        X['%K'] =  ticker['%K'] 
        X['%D'] =  ticker['%D']
        X['14-high'] = ticker['14-high'] 
        X['14-low'] = ticker['14-low'] 
        return X
    
    
    
    
    
def stochastic_oscillator(ticker):
    ticker['14-high'] = ticker['high'].rolling(14).max()
    ticker['14-low'] = ticker['low'].rolling(14).min()
    ticker['%K'] = (ticker['close'] - ticker['14-low'])*100/(ticker['14-high'] - ticker['14-low'])
    ticker['%D'] = ticker['%K'].rolling(3).mean()
    return ticker

def plot_stoch(symbol, price, k, d):
    ax1 = plt.subplot2grid((9, 1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((9, 1), (6,0), rowspan = 3, colspan = 1)
    ax1.plot(price)
    ax1.set_title(f'{symbol} STOCK PRICE')
    ax2.plot(k, color = 'deepskyblue', linewidth = 1.5, label = '%K')
    ax2.plot(d, color = 'orange', linewidth = 1.5, label = '%D')
    ax2.axhline(80, color = 'black', linewidth = 1, linestyle = '--')
    ax2.axhline(20, color = 'black', linewidth = 1, linestyle = '--')
    ax2.set_title(f'{symbol} STOCH')
    ax2.legend()
    plt.show()
    
    
def implement_stoch_strategy(prices, k, d):    
    buy_price = []
    sell_price = []
    stoch_signal = []
    signal = 0

    for i in range(len(prices)):
        if k[i] < 20 and d[i] < 20 and k[i] < d[i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                stoch_signal.append(0)
        elif k[i] > 80 and d[i] > 80 and k[i] > d[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                stoch_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                stoch_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            stoch_signal.append(0)
            
    return buy_price, sell_price, stoch_signal


def plot_sell_buy(ticker, buy_price, sell_price, stoch_signal):
    ax1 = plt.subplot2grid((9, 1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((9, 1), (6,0), rowspan = 3, colspan = 1)
    ax1.plot(ticker['close'], color = 'skyblue', label = 'AALP')
    ax1.plot(ticker.index, buy_price, marker = '^', color = 'green', markersize = 10, label = 'BUY SIGNAL', linewidth = 0)
    ax1.plot(ticker.index, sell_price, marker = 'v', color = 'r', markersize = 10, label = 'SELL SIGNAL', linewidth = 0)
    ax1.legend(loc = 'upper left')
    ax1.set_title('AALP STOCK PRICE')
    ax2.plot(ticker['%K'], color = 'deepskyblue', linewidth = 1.5, label = '%K')
    ax2.plot(ticker['%D'], color = 'orange', linewidth = 1.5, label = '%D')
    ax2.axhline(80, color = 'black', linewidth = 1, linestyle = '--')
    ax2.axhline(20, color = 'black', linewidth = 1, linestyle = '--')
    ax2.set_title('AALP STOCH')
    ax2.legend()
    plt.show()