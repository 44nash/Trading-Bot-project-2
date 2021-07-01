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
