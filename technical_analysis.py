import pandas as pd
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


def main() -> None:
    return None


if __name__ == '__main__':
    main()
