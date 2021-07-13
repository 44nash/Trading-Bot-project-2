# Import the usual suspects....
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


import pandas_datareader as pdr
import datetime as dt
import quandl


import pandas as pd
import numpy as np
import requests
from termcolor import colored as cl
from math import floor
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')


import os
import random
import numpy as np
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'



import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os
import pandas as pd





def get_api_df():
    # https://algotrading101.com/learn/alpaca-trading-api-guide/
    # "C:\\Users\\thebe\\Fintech\\API_KEYS.env"
    # "../../api_keys.env"
    load_dotenv("../../../api_keys.env")

    # authentication and connection details
    # print(os.getenv("ALPACA_API_KEY"))
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")
    base_url = 'https://paper-api.alpaca.markets'

    # instantiate REST API
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    # obtain account information
    account = api.get_account()
    # print(account)


    today = pd.Timestamp("2020-07-14", tz="America/New_York").isoformat()

    # tickers = ["FB", "TWTR", 'AAPL']
    tickers = ['AAPL']
    # tickers =["FB"]

    timeframe = "1D"



    start = pd.Timestamp("2017-07-2", tz="America/New_York").isoformat()
    end = pd.Timestamp("2021-07-2", tz="America/New_York").isoformat()

    # Get current closing prices for FB and TWTR
    ticker0 = api.get_barset(
        tickers,
        timeframe,
        start = start,
        end = end,
    ).df

    ticker = ticker0.copy()
    ticker.columns= ticker0["AAPL"].columns.str.strip().str.lower()


    # Display sample data
    ticker.head(5)
    return ticker
