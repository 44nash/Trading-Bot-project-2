import alpaca_trade_api as tradeapi
from alpaca_trade_api import StreamConn
import threading
import time
import datetime
import logging
import argparse
# You must initialize logging, otherwise you'll not see debug output.
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

import os
from dotenv import load_dotenv
import random
import numpy as np
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

from alpaca_trade_api import StreamConn
from alpaca_trade_api.common import URL



load_dotenv("../../../../api_keys.env")

# pip install polygon-api-client

# API KEYS
#  https://app.alpaca.markets/paper/dashboard/overview
#region
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"

POLYGON_KEY = os.getenv("POLYGON_KEY")
polygon_key2 = os.getenv("evil_einstein")



#endregion
#Buy a stock when a doji candle forms
class Bot:
    def __init__(self, symbol: str = 'AAPL', ):
        self.alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, api_version='v2')
        self.symbol = symbol
    
    def run(self, target_price, buy_sell):
        #On Each Minute
        async def on_minute(conn, channel, bar):
            symbol = bar.symbol
            print("Close: ", bar.close)
            print("Open: ", bar.open)
            print("Low: ", bar.low)
            print(symbol)
            #Check for Doji
            if bar.close > bar.open and bar.open - bar.low > 0.1:
                print('Buying on Doji!')
                self.alpaca.submit_order(symbol,1,'buy','market','day')
            #TODO : Take profit

            
        # Connect to get streaming market data
        # conn = StreamConn('Polygon Key Here', 'Polygon Key Here', 'wss://alpaca.socket.polygon.io/stocks')
        # 23oh7xy7ZoipBODtz9NTPh8B7z9EZrPG    
        
        conn = StreamConn(POLYGON_KEY , POLYGON_KEY , 'wss://alpaca.socket.polygon.io/stocks')
        on_minute = conn.on(r'AM$')(on_minute)
        # Subscribe to Microsoft Stock 
        conn.run([f'AM.{self.symbol}'])

        
        
        
        
        
        
    def hello(self):
        return "hello"
        
    def get_position(self):
        openPosition = self.alpaca.get_position(self.symbol)
        print(openPosition)
        
    def buy(self):
        targetPositionSize = 1
        returned = self.alpaca.submit_order(self.symbol,targetPositionSize,"buy","market","gtc") # Market order to open position
        print(returned)

    def sell(self):
        openPosition = api.get_position(self.symbol)
        returned = self.alpaca.submit_order.submit_order(self.symbol,openPosition,"sell","market","gtc") # Market order to fully close position
        print(returned)
    
    def check_symbol_price(self):
        
        async def on_minute(conn, channel, bar):
            symbol = self.symbol
            print("Close/ Current price: ", bar.close)
            print("Open: ", bar.open)
            print("Low: ", bar.low)
            print(symbol)
        
        conn = StreamConn(POLYGON_KEY , POLYGON_KEY , 'wss://alpaca.socket.polygon.io/stocks')
        on_minute = conn.on(r'AM$')(on_minute)
        # Subscribe to Microsoft Stock 
        # conn.run([f'AM.{self.symbol}'])
        conn.run(['AM.MSFT'])
            
    def check(self):
        conn = StreamConn(
            API_KEY,
            API_SECRET,
            base_url=URL('wss://paper-api.alpaca.markets'),
            data_url=URL('https://data.alpaca.markets'),
            data_stream='alpacadatav1'
        )

        @conn.on(r'Q\..+')
        async def on_quotes(conn, channel, quote):
            print('quote', quote)
            print("Close/ Current price: ", quote.close)
            print("Open: ", quote.open)
            print("Low: ", quote.low)


        conn.run(['alpacadatav1/Q.GOOG'])





        
        
        
# Run the BuyDoji class
# ls = Bot()
# ls.hello()
# ls.run()