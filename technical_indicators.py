"""
TODO
----

# Technical Indicators

- [x] MACD
- [x] RSI
- [ ] Average True Range
- [ ] Rolling mean and standard deviation
- [ ] Bollinger Bands
- [ ] Average Directional Index
- [ ] Stochastic
- [ ] Add names to `pd.Series` that are returned? or return a `pd.DataFrame` instead?
- [ ] `if calc_method == 'EMA': calc_method = ema_calc_methods_default`

Notes
-----

## Exponential Moving Average (EMA)

See [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
for documentation on the four different EMA calculation types implemented in
pandas: `{'alpha', 'com', 'halflife', 'span'}`. Note: The N-day SMA has a lag of
(N - 1)/2; To make the EMA have the same lag as the N-day SMA, choose
`calc_method = 'span' (this is the most commonly used setting in stock-market
technical analysis).

"""
import numpy as np
import pandas as pd
from typing import Tuple, Union


# Allowed exponential-moving-average (EMA) calculation methods.  See
# [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
# for more details.
ema_calc_methods = {'alpha', 'com', 'halflife', 'span'}
ema_calc_methods_default = 'span'


def get_sma(
    timeseries: Union[pd.Series, pd.DataFrame] = None,
    window: int = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the simple moving average (SMA) of an input
    timeseries or dataframe.
    """
    return timeseries.rolling(window=window, min_periods=window).mean()


def get_ema(
    timeseries: Union[pd.Series, pd.DataFrame] = None,
    window: int = None,
    calc_method: str = 'span',
) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the exponential moving average (EMA) of an
    input timeseries or dataframe.

    See [pandas.DataFrame.ewm](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    for information about the four different `calc_methods`, i.e. {'alpha',
    'com', 'halflife', 'span'}.

    The N-day SMA has a lag of (N - 1)/2.  To make the EMA have the same lag
    as the N-day SMA, choose `calc_method = 'span' (this is the most commonly
    used setting in stock-market technical analysis).
    """
    # Check user's arguments
    if calc_method.lower() in ema_calc_methods:
        calc_method = calc_method.lower()
    else:
        calc_method = ema_calc_methods_default

    # Calculate EMA and return the result
    kwargs = {calc_method.lower(): window, 'min_periods': window}
    return timeseries.ewm(**kwargs).mean()


def get_dema(
    timeseries: Union[pd.Series, pd.DataFrame] = None,
    window: int = None,
    calc_method: str = 'span',
) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the double exponential moving average (DEMA)
    of an input timeseries or dataframe.

    See [pandas.DataFrame.ewm](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    for information about the four different `calc_methods`, i.e. {'alpha',
    'com', 'halflife', 'span'}.

    The N-day SMA has a lag of (N - 1)/2.  To make the EMA have the same lag
    as the N-day SMA, choose `calc_method = 'span' (this is the most commonly
    used setting in stock-market technical analysis).
    """
    kwargs = dict(window=window, calc_method=calc_method)
    ema1_vals = get_ema(timeseries, **kwargs)
    ema2_vals = get_ema(ema1_vals, **kwargs)
    return (2 * ema1_vals) - ema2_vals


def get_tema(
    timeseries: Union[pd.Series, pd.DataFrame] = None,
    window: int = None,
    calc_method: str = 'span',
) -> Union[pd.Series, pd.DataFrame]:
    """
    Helper function to calculate the triple exponential moving average (TEMA)
    of an input timeseries or dataframe.

    See [pandas.DataFrame.ewm](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    for information about the four different `calc_methods`, i.e. {'alpha',
    'com', 'halflife', 'span'}.

    The N-day SMA has a lag of (N - 1)/2.  To make the EMA have the same lag
    as the N-day SMA, choose `calc_method = 'span' (this is the most commonly
    used setting in stock-market technical analysis).
    """
    kwargs = dict(window=window, calc_method=calc_method)
    ema1_vals = get_ema(timeseries, **kwargs)
    ema2_vals = get_ema(ema1_vals, **kwargs)
    ema3_vals = get_ema(ema2_vals, **kwargs)
    return (3 * ema1_vals) - (3 * ema2_vals) + ema3_vals


def get_macd(
    timeseries: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    smoothing_period: int = 9,
    calc_method: str = ema_calc_methods_default,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper function to calculate the moving-average convergence/divergence
    (MACD) of an input timeseries.

    Default parameter and return values taken from [StockCharts.com](https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd).
    """
    # Here's what we're calculating
    macd_vals = None
    macd_signal_vals = None
    macd_hist_vals = None

    # Calculate the moving-average convergence/divergence (MACD)
    if calc_method.upper() == 'SMA':
        # Use simple moving average (SMA)
        ma_fast_vals = get_sma(timeseries, window=fast_period)
        ma_slow_vals = get_sma(timeseries, window=slow_period)
        macd_vals = ma_fast_vals - ma_slow_vals
        macd_signal_vals = get_sma(macd_vals, window=smoothing_period)
    elif calc_method.lower() in ema_calc_methods:
        # Use exponential moving average (EMA)
        kwargs = dict(calc_method=calc_method)
        ma_fast_vals = get_ema(timeseries, window=fast_period, **kwargs)
        ma_slow_vals = get_ema(timeseries, window=slow_period, **kwargs)
        macd_vals = ma_fast_vals - ma_slow_vals
        macd_signal_vals = get_ema(macd_vals, window=smoothing_period, **kwargs)
    else:
        raise ValueError(
            f"EMA calculation method `calc_method = {calc_method}` not "
            f"recognized!")

    # Calculate MACD histogram values
    macd_hist_vals = macd_vals - macd_signal_vals

    # Return the results
    return (macd_vals, macd_signal_vals, macd_hist_vals)


def get_rsi(
    timeseries: pd.Series,
    lookback: int = 14,
    diff_period: int = 1,
) -> pd.Series:
    """
    Helper function to calculate the relative strength index (RSI).
    """
    # Here's what we're calculating
    rsi_vals = np.full_like(timeseries, np.nan)

    # Cache difference values
    diff_vals = timeseries.diff(periods=diff_period)
    diff_vals[0] = 0.0  # replace `NaN`

    # Loop over data
    for idx_end in range(lookback, (diff_vals.size + 1)):
        idx_begin = idx_end - lookback
        diff_vals_slice = diff_vals[idx_begin:idx_end]
        # Calculate RSI for this slice
        up_sum = abs(diff_vals_slice[np.where(diff_vals_slice > 0.0, True, False)].sum())
        dn_sum = abs(diff_vals_slice[np.where(diff_vals_slice < 0.0, True, False)].sum())
        rsi_vals[idx_end - 1] = 100 * (up_sum / (up_sum + dn_sum))

    # Return the results
    return pd.Series(data=rsi_vals, index=timeseries.index, name=f"RSI{lookback}")


def get_bollinger_bands(
    timeseries: pd.Series,
    lookback: int = 20,
    num_stdevs: float = 2.0,
    calc_method: str = 'SMA',
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Helper function to calculate the Bollinger Bands of an input timeseries.
    """
    # Here's what we're calculating
    bbands_up = None
    bbands_ct = None
    bbands_dn = None

    # Cache rolling window of input timeseries
    if calc_method.upper() == 'SMA':
        # For use with simple moving average (SMA) -- see below
        rolling = timeseries.rolling(window=lookback, min_periods=lookback)
    else:
        # For use with exponential moving average (EMA) -- see below
        rolling = timeseries.ewm(
            **{calc_method.lower(): lookback, 'min_periods': lookback})

    # Cache rolling standard deviation of the price
    rolling_stdev = rolling.std()

    # Calculate Bollinger Bands
    bbands_ct = rolling.mean()
    bbands_up = bbands_ct + (num_stdevs * rolling_stdev)
    bbands_dn = bbands_ct - (num_stdevs * rolling_stdev)

    # Return the results
    return (bbands_up, bbands_ct, bbands_dn)


get_bbands = get_bollinger_bands  # shorthand


def get_true_range(
    df_hlc: pd.DataFrame,
    high_label: str = 'High',
    low_label: str = 'Low',
    close_label: str = 'Close',
) -> pd.Series:
    """
    Helper function to calculate the True Range (TR) of the price values
    contained in an input dataframe of {High, Low, Close} price values.
    """
    # Cache previous-close values (for use below)
    prev_close_vals = df_hlc[close_label].shift(periods=1)

    # Assemble the 3 components needed to calculate the True Range
    true_range_vals_1 = df_hlc[high_label] - df_hlc[low_label]
    true_range_vals_2 = (df_hlc[high_label] - prev_close_vals).abs()
    true_range_vals_3 = (df_hlc[low_label] - prev_close_vals).abs()

    # Calculate the True Range
    true_range_vals = np.max(np.vstack([true_range_vals_1,
                                        true_range_vals_2,
                                        true_range_vals_3]), axis=0)

    # Convert to `pandas.Series` and return
    return pd.Series(data=true_range_vals, index=df_hlc.index, name='TrueRange')


def get_average_true_range(
    df_hlc: pd.DataFrame,
    lookback: int = 14,
    calc_method: str = 'SMA',
    high_label: str = 'High',
    low_label: str = 'Low',
    close_label: str = 'Close',
) -> pd.Series:
    """
    Helper function to calculate the Average True Range (TR) of the price
    values contained in an input dataframe of {High, Low, Close} price values.
    """
    # Check user's arguments
    if calc_method.upper() == 'EMA':
        calc_method = ema_calc_methods_default

    # Cache true-range (TR) values (for use below)
    true_range_vals = get_true_range(df_hlc,
                                     high_label=high_label,
                                     low_label=low_label,
                                     close_label=close_label)

    # Calculate average true range (ATR)
    if calc_method.upper() == 'SMA':
        # Simple moving average (SMA)
        atr_vals = get_sma(true_range_vals, window=lookback)
    elif calc_method.lower() in ema_calc_methods:
        # Exponential moving average (EMA)
        atr_vals = get_ema(true_range_vals, window=lookback,
                           calc_method=calc_method)
    else:
        # Problem
        raise ValueError(f"`calc_method = {calc_method}` not recognized!")

    # Label the output `pd.Series`
    atr_vals.name = f"ATR{lookback}"

    # Return the results
    return atr_vals


get_atr = get_average_true_range  # shorthand


def get_adx_dis(
    df_hlc: pd.DataFrame,
    lookback: int = 14,
    calc_method: str = 'EMA',
    high_label: str = 'High',
    low_label: str = 'Low',
    close_label: str = 'Close',
):
    """
    Helper function to calculate the average directional index (ADX) and
    directional indicators (DI+, DI-) from an input dataframe of {High, Low,
    Close} price vals.

    Note: The convention for the choice of exponential rate of decay varies
    somewhat, e.g. `math`:alpha = 1 / N: versus `math`:2 / (N + 1):.
    """
    # Here's what we're calculating
    adx_vals = None
    di_plus_vals = None
    di_minus_vals = None

    # Cache high and low values
    high_vals = df_hlc[high_label].to_numpy()
    low_vals = df_hlc[low_label].to_numpy()

    # Calculate directional movement (DM)
    index = df_hlc.index
    dm_plus_vals = np.zeros(index.size)
    dm_minus_vals = np.zeros(index.size)
    for idx in range(1, index.size):
        dm_plus = high_vals[idx] - high_vals[idx - 1]
        dm_minus = low_vals[idx - 1] - low_vals[idx]
        if (dm_plus > 0.0) and (dm_plus > dm_minus):
            dm_plus_vals[idx] = dm_plus
        if (dm_minus > 0.0) and (dm_minus > dm_plus):
            dm_minus_vals[idx] = dm_minus

    # Get average true range (ATR) for normalizing DM (above)
    atr_vals = get_average_true_range(
        df_hlc, lookback=lookback, calc_method=calc_method,
        high_label=high_label, low_label=low_label, close_label=close_label)

    # Calculate directional indicators (DIs) and average directional index (ADX)
    if calc_method.upper() == 'SMA':
        # Use simple moving average (SMA) to calculate ADX/DIs
        di_plus_vals = 100 * get_sma((dm_plus_vals / atr_vals), window=lookback)
        di_minus_vals = 100 * get_sma((dm_minus_vals / atr_vals), window=lookback)
        dx_vals = 100 * np.abs(di_plus_vals - di_minus_vals) / (di_plus_vals + di_minus_vals)
        adx_vals = get_sma(dx_vals, window=lookback)
    else:
        # Use exponential moving average (EMA) to calculate ADX/DIs
        di_plus_vals = 100 * get_ema((dm_plus_vals / atr_vals), window=lookback, calc_method=calc_method)
        di_minus_vals = 100 * get_ema((dm_minus_vals / atr_vals), window=lookback, calc_method=calc_method)
        dx_vals = 100 * np.abs(di_plus_vals - di_minus_vals) / (di_plus_vals + di_minus_vals)
        adx_vals = get_ema(dx_vals, window=lookback, calc_method=calc_method)
    
    # Label output timeseries
    adx_vals.name = 'ADX'
    di_plus_vals.name = 'DI+'
    di_minus_vals.name = 'DI-'

    # Return the results
    return adx_vals, di_plus_vals, di_minus_vals


def main() -> None:
    return None


if __name__ == '__main__':
    main()
