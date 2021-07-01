import pandas as pd
from typing import Union


def get_sma(timeseries: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the simple moving average (SMA) of an input
    timeseries.
    """
    return timeseries.rolling(window=window, min_periods=window).mean()


def main() -> None:
    return None


if __name__ == '__main__':
    main()
