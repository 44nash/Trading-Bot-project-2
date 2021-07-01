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


def main() -> None:
    return None


if __name__ == '__main__':
    main()
