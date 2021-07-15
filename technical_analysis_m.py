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
    
    
    ma_types_ = {'stochastic oscillator', 'ichimoku cloud', 'fibonacci retracement'}
    
    def __init__( 
        self,
        ma_type: str = 'stochastic oscillator',
        window: int = 20,
        input_label: str = 'close',
        output_label: str = None,
                ):
        # Check user's arguments
        self.ma_type = ma_type.lower()
        if not (self.ma_type in self.ma_types_):
            raise ValueError(f"ERROR: Moving-average type \'{self.ma_type}\' "
                             f"not in {self.ma_types_}!")
            
        # Assign member data
        self.window = window
        self.input_label = input_label
        if output_label is None:
            self.output_label = f"{self.ma_type}{self.window}"

        
    def fit(self, X: pd.DataFrame, **fit_params):
        return self  # nothing to do!
    
    def transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        if self.ma_type == 'stochastic oscillator':
            ticker = stochastic_oscillator(X)
            X['%K'] =  ticker['%K'] 
            X['%D'] =  ticker['%D']
            X['14-high'] = ticker['14-high'] 
            X['14-low'] = ticker['14-low']
        elif self.ma_type == 'ichimoku cloud':
            ichimoku_dataframe(X,  view_limit=100)
        elif self.ma_type == 'fibonacci retracement':
             get_fib_retracement_levels(X)
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
    
    
    
    
    
# Ichimoku Kinko Hyo Cloud



tenkan_window = 20
kijun_window = 60
senkou_span_b_window = 120
cloud_displacement = 30
chikou_shift = -30

def kijun_sen(df,kijun_window ):

    # Kijun 
    kijun_sen_high = df['high'].rolling( window=kijun_window ).max()
    kijun_sen_low = df['low'].rolling( window=kijun_window ).min()
    df['kijun_sen'] = (kijun_sen_high + kijun_sen_low) / 2

    return df

def kijun_sen_plot(ticker,kijun_window):
    ticker = kijun_sen(ticker,kijun_window)
    # plt.plot(ticker[-250:, Close], color = 'black', label = 'EURUSD')
    plt.plot(ticker["close"], color = 'black',label = "AAPL")
    # plt.plot(ticker[-250:, where], color = 'blue', label = 'Kijun-Sen')
    plt.plot(ticker["kijun_sen"], color = 'blue',label = "Kijun-Sen")
    plt.grid()
    plt.legend()
    
    
    
    
    
    
    
    
    
    
    
    
    
def tenkan_sen(df,tenkan_window ):
    # Tenkan 
    tenkan_sen_high = df['high'].rolling( window=tenkan_window ).max()
    tenkan_sen_low = df['low'].rolling( window=tenkan_window ).min()
    df['tenkan_sen'] = (tenkan_sen_high + tenkan_sen_low) /2
    
    return df

def tenkan_sen_plot(ticker, tenkan_window ):
    ticker = tenkan_sen(ticker, tenkan_window )
    plt.plot(ticker["close"], color = 'black', label = 'AAPL')
    plt.plot(ticker['tenkan_sen'], color = 'red', label = 'Tenkan-Sen')
    plt.grid()
    plt.legend()
    
    
    
    
    
    
    
    


def Senkou_Span_A(df, cloud_displacement ):
    # Senkou Span A 
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(cloud_displacement)
    return df

def Senkou_Span_B(df, senkou_span_b_window):
    # Senkou Span B 
    senkou_span_b_high = df['high'].rolling( window=senkou_span_b_window ).max()
    senkou_span_b_low = df['low'].rolling( window=senkou_span_b_window ).min()
    df['senkou_span_b'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(cloud_displacement)
    return df

def Senkou_Span_plot(ticker, senkou_span_b_window):
    ticker_span_b = Senkou_Span_B(ticker, senkou_span_b_window)
    ticker_span_a = Senkou_Span_A(ticker, cloud_displacement )
    plt.plot(ticker["close"] , color = 'black', label = 'AAPL')
    plt.plot(ticker_span_b['senkou_span_a'] , color = 'yellow', label = 'senkou_span_a')
    plt.plot(ticker_span_b['senkou_span_b'] , color = 'cyan', label = 'senkou_span_b')
    plt.grid()
    plt.legend()








def Chikou(df, chikou_shift):
    # Chikou
    df['chikou_span']  = df["close"].shift(chikou_shift)
    return df
def Chikou_plot(ticker, chikou_shift):
    ticker = Chikou(ticker, chikou_shift)
    plt.plot(ticker["close"] , color = 'black', label = 'AAPL')
    plt.plot(ticker['chikou_span'] , color = 'green', label = 'Chikou-Span')
    plt.grid()
    plt.legend()






def ichimoku_dataframe(ticker,  view_limit=100): 
    
    ticker = kijun_sen(ticker,kijun_window)
    ticker = tenkan_sen(ticker, tenkan_window )
    ticker = Chikou(ticker, chikou_shift)
    ticker = Senkou_Span_B(ticker, senkou_span_b_window)
    ticker = Senkou_Span_A(ticker, cloud_displacement )
    
    return ticker
    

# fig, ax = plt.subplots() 
def plot_ichimoku(ticker,  view_limit=100): 
    
    ticker0 = kijun_sen(ticker,kijun_window)
    ticker1 = tenkan_sen(ticker0, tenkan_window )
    ticker2 = Chikou(ticker1, chikou_shift)
    ticker3 = Senkou_Span_B(ticker2, senkou_span_b_window)
    ticker4 = Senkou_Span_A(ticker3, cloud_displacement )
    
    df = ticker4
    
    d2 = df.loc[:, ['tenkan_sen','kijun_sen','senkou_span_a','senkou_span_b', 'chikou_span']]
    d2 = d2.tail(view_limit)
    date_axis = d2.index.values
    # ichimoku
    plt.plot(date_axis, d2['tenkan_sen'], label="tenkan", color='#0496ff', alpha=0.65,linewidth=1)
    plt.plot(date_axis, d2['kijun_sen'], label="kijun", color="#991515", alpha=0.65,linewidth=1)
    plt.plot(date_axis, d2['senkou_span_a'], label="span a", color="#008000", alpha=0.65,linewidth=1)
    plt.plot(date_axis, d2['senkou_span_b'], label="span b", color="#ff0000", alpha=0.65, linewidth=1)
    plt.plot(date_axis, d2['chikou_span'], label="chikou", color="#ffffff", alpha=0.65, linewidth=1)
    # green cloud
    plt.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'], where=d2['senkou_span_a']> d2['senkou_span_b'], facecolor='#008000', interpolate=True, alpha=0.25)
    # red cloud
    plt.fill_between(date_axis, d2['senkou_span_a'], d2['senkou_span_b'], where=d2['senkou_span_b']> d2['senkou_span_a'], facecolor='#ff0000', interpolate=True, alpha=0.25)
    
    
    
    
    
# Fibonacci retracement


def get_fib_retracement_levels(data, verbosity: int = 0):
    """
    Helper function to calculate the fibonacci retracement levels
    """
    #Fibonacci retracement levels ratios
    retracement_levels = [0.236, 0.382, 0.5 , 0.618]
    
    #min and max of Close price
    closePriceMax = data['close'].max()
    closePriceMin = data['close'].min()
    
    #the difference between max and min Close Price (total up/down move)
    diff = closePriceMax - closePriceMin
    
    
    #calculation of price per retracement levels ratios
    level_1 = closePriceMax - retracement_levels[0] * diff
    level_2 = closePriceMax - retracement_levels[1] * diff
    level_3 = closePriceMax - retracement_levels[2] * diff
    level_4 = closePriceMax - retracement_levels[3] * diff
    
    #Print the price at each level
    if verbosity > 0:
        print("Level Percentage\t", "Price ($)")
        print("00.0%\t\t", closePriceMax)
        print("23.6%\t\t", level_1)
        print("38.2%\t\t", level_2)
        print("50.0%\t\t", level_3)
        print("61.8%\t\t", level_4)
        print("100.0%\t\t", closePriceMin)
    
    data['fib_close_min'] = closePriceMin
    data['fib_level_1'] = level_1
    data['fib_level_2'] = level_2
    data['fib_level_3'] = level_3
    data['fib_level_4'] = level_4
    data['fib_close_max'] = closePriceMax

    
    
#     return data, closePriceMin, level_1, level_2, level_3, level_4, closePriceMax
    return data







# This is how you call it 
# fib_retracement_plot(
#     ticker,
#     ticker['fib_close_min'][0],
#     ticker['fib_level_1'][0],
#     ticker['fib_level_2'][0], 
#     ticker['fib_level_3'][0],
#     ticker['fib_level_4'][0],
#     ticker['fib_close_max'][0])
def fib_retracement_plot(df, closePriceMin, level_1, level_2, level_3, level_4, closePriceMax):

    new_df = df
    plt.figure(figsize=(12.33,4.5))
#     ax = fig.add_subplot(1,1,1)
    
    plt.title('Fibonnacci Retracement Plot')
    plt.plot(new_df.index, new_df['close'])
    
    plt.axhline(closePriceMax, linestyle='--', alpha=0.5, color = 'red')
    plt.fill_between(df.index, closePriceMax, level_1, color = 'red')
    
    plt.axhline(level_1, linestyle='--', alpha=0.5, color = 'orange')
    plt.fill_between(df.index, level_1, level_2, color = 'orange')
    
    plt.axhline(level_2, linestyle='--', alpha=0.5, color = 'yellow')
    plt.fill_between(df.index, level_2, level_3, color = 'yellow')
    
    plt.axhline(level_3, linestyle='--', alpha=0.5, color = 'green')
    plt.fill_between(df.index, level_3, level_4, color = 'green')
    
    plt.axhline(level_4, linestyle='--', alpha=0.5, color = 'blue')
    plt.fill_between(df.index, level_4, closePriceMin, color = 'blue')
    
    plt.axhline(closePriceMin, linestyle='--', alpha=0.5, color = 'purple')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price in USD',fontsize=18)
    plt.show()