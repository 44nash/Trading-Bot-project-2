# Project 2: Algorithmic Trading Bot

Team Members:

- Benjamin Adler
- Emmanuel Constant
- Nathan Froemming, Ph.D.
- Marcus Nash

## Introduction

In this project, we will create an algorithmic-trading system by combining technical analysis of timeseries, machine-learning algorithms, and application programming interfaces (APIs) for trading stocks, bonds, and cryptocurrency.  The input data will consist of {Open, High, Low, Close, Volume} (OHLCV) data freely available from [Alpaca](https://alpaca.markets/docs/api-documentation/how-to/market-data/) and [Kraken](https://docs.kraken.com/rest/#operation/getOHLCData) APIs.  [Technical indicators](https://www.ig.com/us/trading-strategies/10-trading-indicators-every-trader-should-know-190604) will be constructed from the input OHLCV data and transformed into machine-learning features using a [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).  Output/Target variables for the machine-learning algorithms will consist of 1-day, 5-day, and 10-day returns, for example.  [Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html) will be used to combine the predictions of several machine-learning algorithms (e.g. random forest, recurrent neural networks, etc.) into a single predictor in order to improve generalizability and robustness over using a single algorithm.  Backtesting on historical data will allow for key performance parameters such as profit & loss, length of time positions are held, maximum drawdown, etc., to be assessed.  If time permits, we will also attempt to incorporate alternative data such as [sentiment analysis from FinnHub](https://finnhub.io/docs/api/news-sentiment) into the algorithmic-trading system.  

## References

- [10 Trading Indicators Every Trader Should Know](https://www.ig.com/us/trading-strategies/10-trading-indicators-every-trader-should-know-190604)
