import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union
def get_sma(timeseries: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the simple moving average (SMA) of an input
    timeseries.
    """
    return timeseries.rolling(window=window, min_periods=window).mean()
def get_ema(timeseries: Union[pd.Series, pd.DataFrame] = None,
            window: float = None,
            calc_method: str = 'span') -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the exponential moving average (EMA) of an
    input timeseries.
    See [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html?highlight=ewm#pandas.DataFrame.ewm)
    for information about the four `calc_methods` {'alpha', 'com', 'halflife',
    'span'}.
    """
    kwargs = {calc_method.lower(): window, 'min_periods': window}
    return timeseries.ewm(**kwargs).mean()


def get_dema(timeseries: Union[pd.Series, pd.DataFrame] = None,
             window: float = None,
             calc_method: str = 'span') -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the double exponential moving average (DEMA)
    of an input timeseries.
    See [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html?highlight=ewm#pandas.DataFrame.ewm)
    for information about the four `calc_methods` {'alpha', 'com', 'halflife',
    'span'}.
    """
    kwargs = dict(window=window, calc_method=calc_method)
    ema1_vals = get_ema(timeseries, **kwargs)
    ema2_vals = get_ema(ema1_vals, **kwargs)
    return (2 * ema1_vals) - ema2_vals


def get_tema(timeseries: Union[pd.Series, pd.DataFrame] = None,
             window: float = None,
             calc_method: str = 'span') -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the triple exponential moving average (TEMA)
    of an input timeseries.
    See [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html?highlight=ewm#pandas.DataFrame.ewm)
    for information about the four `calc_methods` {'alpha', 'com', 'halflife',
    'span'}.
    """
    kwargs = dict(window=window, calc_method=calc_method)
    ema1_vals = get_ema(timeseries, **kwargs)
    ema2_vals = get_ema(ema1_vals, **kwargs)
    ema3_vals = get_ema(ema2_vals, **kwargs)
    return (3 * ema1_vals) - (3 * ema2_vals) + ema3_vals


def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns =     {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames = [macd, signal, hist]
    df = pd.concat(frames, join = 'inner', axis = 1)
    return df


def get_rsi(close, lookback):
    ret = close.diff()
    up = []
    down = []
    for i in range(len(ret)):
        if ret[i] < 0:
            up.append(0)
            down.append(ret[i])
        else:
            up.append(ret[i])
            down.append(0)
    up_series = pd.Series(up)
    down_series = pd.Series(down).abs()
    up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
    down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
    rs = up_ewm/down_ewm
    rsi = 100 - (100 / (1 + rs))
    rsi_df = pd.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
    rsi_df = rsi_df.dropna()
    return rsi_df[3:]


def get_bollinger(data, window):
    '''
    Function that creates bollinger bands for a given timeseries
    '''
    sma = data.rolling(window = window).mean()
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb, sma


class BennyIndicators(BaseEstimator, TransformerMixin):

    indicators = {'bolinger_bands', 'macd', 'rsi'}
    
    def __init__(
        self,
        bb_window
        ):
        self.bb_window = bb_window
    
    def fit(self, X: pd.DataFrame, **fit_params):
        return self  # nothing to do!
    
    def transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        close_vals = X['close']
        
        # calculate bollinger bands
        upper_bb, lower_bb, sma = get_bollinger(close_vals, self.bb_window)
        X['upper_bb'] = upper_bb
        X['lower_bb'] = lower_bb
        X['middle_bb'] = sma
        
        # calculate rsi
        X['rsi'] = get_rsi(close_vals, 14)
        
        # calculate macd
        # macd = get_macd()
        return X


class MovingAverageAdder(BaseEstimator, TransformerMixin):
    """
    TODO: Documentation.
    """
    ma_types_ = {'sma', 'ema', 'dema', 'tema'}
    def __init__(
        self,
        ma_type: str = 'sma',
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
        if self.ma_type == 'sma':
            X[self.output_label] = get_sma(X[self.input_label], window=self.window)
        elif self.ma_type == 'ema':
            X[self.output_label] = get_ema(X[self.input_label], window=self.window)
        elif self.ma_type == 'dema':
            X[self.output_label] = get_dema(X[self.input_label], window=self.window)
        elif self.ma_type == 'tema':
            X[self.output_label] = get_tema(X[self.input_label], window=self.window)
        return X
def main() -> None:
    return None
if __name__ == '__main__':
    main()