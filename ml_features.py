import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Our custom technical-indicator code
import technical_indicators as ti
from technical_analysis_m import stochastic_oscillator
from technical_analysis_m import get_fib_retracement_levels


class MLFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    Helper class to add machine-learning features to an input dataframe
    containing {Open, High, Low, Close, Volume} (OHLCV) data from the stock
    market.

    The technical indicators implemented below come from the article entitled,
    ["10 trading indicators every trader should know"](https://www.ig.com/us/trading-strategies/10-trading-indicators-every-trader-should-know-190604)
    (accessed 15 Jul 2021).

    The technical indicators are added to the input dataframe of OHLCV data
    in the same order the indicators are presented in the article:

    - [x] Simple Moving Average (SMA)
    - [x] Exponential Moving Average (EMA)
    - [x] Stochastic Oscillator (STOCH)
    - [x] Moving-Average Convergence/Divergence (MACD)
    - [x] Bollinger Bands (BBANDS)
    - [x] Relative Strength Index (RSI)
    - [x] Fibonacci Retracement
    - [ ] Ichimoku Cloud?
    - [x] Standard Deviation (STDEV)
    - [x] Average Directional Index (ADX)

    """

    def __init__(
        self,
        normalize: bool = True,
        open_label: str = 'Open',
        high_label: str = 'High',
        low_label: str = 'Low',
        close_label: str = 'Close',
    ):
        self.normalize = normalize
        self.open_label = open_label
        self.high_label = high_label
        self.low_label = low_label
        self.close_label = close_label
        return None

    def fit(self, X: pd.DataFrame, **fit_params):
        return self

    def transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """
        Helper function to add all technical indicators to the input dataframe
        containing OHLCV data.
        """
        # Cache average true range (ATR) -- used for normalization below
        atr_vals = ti.get_atr(X)

        # Also cache the standard deviation of the closing prices -- used in
        # Bollinger Bands, Fibonacci Retracements, and for standardization.
        close_vals = X[self.close_label]
        close_stdev = close_vals.rolling(window=20).std()

        #=======================================================================
        # 1. SIMPLE MOVING AVERAGES (SMAs) -------------------------------------
        #=======================================================================
        # Add common SMAs: 5, 10, 20, 50, 100, 200
        sma5 = ti.get_sma(X[self.close_label], window=5)
        sma10 = ti.get_sma(X[self.close_label], window=10)
        sma20 = ti.get_sma(X[self.close_label], window=20)
        sma50 = ti.get_sma(X[self.close_label], window=50)
        sma100 = ti.get_sma(X[self.close_label], window=100)
        sma200 = ti.get_sma(X[self.close_label], window=200)
        if self.normalize:
            # Take the difference of the "fast" and "slow" moving averages and
            # divide by the average true range (ATR) to normalize.  Also include
            # a "random-walk factor" (i.e. the term in the square-root function)
            # to include the contribution to volatility due to the difference
            # between fast and slow moving-average lengths.
            X['SMA5*'] = (sma5 - sma10) / (atr_vals * np.sqrt(0.5 * abs(5 - 10)))
            X['SMA10*'] = (sma10 - sma20) / (atr_vals * np.sqrt(0.5 * abs(10 - 20)))
            X['SMA20*'] = (sma20 - sma50) / (atr_vals * np.sqrt(0.5 * abs(20 - 50)))
            X['SMA50*'] = (sma50 - sma100) / (atr_vals * np.sqrt(0.5 * abs(50 - 100)))
            X['SMA100*'] = (sma100 - sma200) / (atr_vals * np.sqrt(0.5 * abs(100 - 200)))
        else:
            X['SMA5'] = sma5
            X['SMA10'] = sma10
            X['SMA20'] = sma20
            X['SMA50'] = sma50
            X['SMA100'] = sma100
            X['SMA200'] = sma200

        #=======================================================================
        # 2. EXPONENTIAL MOVING AVERAGES (EMAs)---------------------------------
        #=======================================================================
        ema10 = ti.get_ema(X[self.close_label], window=10, calc_method='span')
        dema10 = ti.get_dema(X[self.close_label], window=10, calc_method='span')
        ema20 = ti.get_ema(X[self.close_label], window=20, calc_method='span')
        dema20 = ti.get_dema(X[self.close_label], window=20, calc_method='span')
        if self.normalize:
            # Similar normalization as SMAs (see above).  Remember, the lag of
            # an EMA is (N-1)/2, so the lag of a DEMA is (N-1).  This explains
            # why the term in the square-root is slightly different than for
            # the simple moving averages above.
            X['DEMA10*'] = (dema10 - ema10) / (atr_vals * np.sqrt(0.5 * abs(20 - 10)))
            X['DEMA20*'] = (dema20 - ema20) / (atr_vals * np.sqrt(0.5 * abs(40 - 20)))
        else:
            X['EMA10'] = ema10
            X['DEMA10'] = dema10
            X['EMA20'] = ema20
            X['DEMA20'] = dema20

        #=======================================================================
        # 3. STOCHASTIC OSCILLATOR (STOCH) -------------------------------------
        #=======================================================================
        df_stoch = X[[self.high_label, self.low_label, self.close_label]].copy()
        df_stoch = df_stoch.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'})
        df_stoch = stochastic_oscillator(df_stoch)
        stoch_kfast = df_stoch['%K']
        stoch_dfast = df_stoch['%D']
        if self.normalize:
            stoch_kfast_norm = (stoch_kfast - 50) / 50  # recenter to [-1, 1]
            stoch_dfast_norm = (stoch_dfast - 50) / 50
            X['STOCH_KFAST*'] = stoch_kfast_norm
            X['STOCH_HIST*'] = stoch_kfast_norm - stoch_dfast_norm  # TODO: `StandardScaler()`?
        else:
            X['STOCH_KFAST'] = stoch_kfast
            X['STOCH_DFAST'] = stoch_dfast

        #=======================================================================
        # 4. MOVING-AVERAGE CONVERGENCE/DIVERGENCE (MACD) ----------------------
        #=======================================================================
        macd_vals, _, macd_hist_vals = ti.get_macd(X[self.close_label])
        if self.normalize:
            X['MACD*'] = macd_vals / (atr_vals * np.sqrt(0.5 * abs(12 - 26)))
            X['MACD_HIST*'] = macd_hist_vals / (atr_vals * np.sqrt(0.5 * abs(abs(12 - 26) - 9)))
        else:
            X['MACD'] = macd_vals
            X['MACD_HIST'] = macd_hist_vals

        #=======================================================================
        # 5. BOLLINGER BANDS (BBANDS) ------------------------------------------
        #=======================================================================
        bbands_up, bbands_ct, bbands_dn = ti.get_bbands(X[self.close_label])
        if self.normalize:
            # Let's use half the BB channel width as a meaningful measure of
            # price scale.  
            X['BBANDS_ZSCORE*'] = (close_vals - bbands_ct) / (2 * close_stdev)

            # We could also look at the position of the close within the BB channel
            #X['BBANDS_%B*'] = 2 * ((close_vals - bbands_dn) / (bbands_up - bbands_dn)) - 1  # [-1, 1]
        else:
            X['BBANDS_UP'] = bbands_up
            X['BBANDS_CT'] = bbands_ct
            X['BBANDS_DN'] = bbands_dn

        #=======================================================================
        # 6. RELATIVE STRENGTH INDEX (RSI) -------------------------------------
        #=======================================================================
        rsi_vals = ti.get_rsi(X[self.close_label])
        if self.normalize:
            X['RSI*'] = (rsi_vals - 50) / 50
        else:
            X['RSI'] = rsi_vals

        #=======================================================================
        # 7. FIBONACCI RETRACEMENT ---------------------------------------------
        #=======================================================================
        df_fib = X[self.close_label].copy().to_frame().rename(columns={'Close': 'close'})
        df_fib = get_fib_retracement_levels(df_fib)
        if self.normalize:
            fib_range = 0.5 * (df_fib['fib_close_max'] - df_fib['fib_close_min']).abs()
            X['FIB_MIN*'] = (close_vals - df_fib['fib_close_min']) / fib_range
            X['FIB_236*'] = (close_vals - df_fib['fib_level_1']) / fib_range 
            X['FIB_382*'] = (close_vals - df_fib['fib_level_2']) / fib_range
            X['FIB_500*'] = (close_vals - df_fib['fib_level_3']) / fib_range
            X['FIB_618*'] = (close_vals - df_fib['fib_level_4']) / fib_range
            X['FIB_MAX*'] = (close_vals - df_fib['fib_close_max']) / fib_range
        else:
            X['FIB_MIN'] = df_fib['fib_close_min']
            X['FIB_236'] = df_fib['fib_level_1']
            X['FIB_382'] = df_fib['fib_level_2']
            X['FIB_500'] = df_fib['fib_level_3']
            X['FIB_618'] = df_fib['fib_level_4']
            X['FIB_MAX'] = df_fib['fib_close_max']

        #=======================================================================
        # 8. ICHIMOKU CLOUD ----------------------------------------------------
        #=======================================================================
        if self.normalize:
            pass
        else:
            pass

        #=======================================================================
        # 9. STANDARD DEVIATION (STDEV) ----------------------------------------
        #=======================================================================
        X['STDEV'] = close_stdev
        X['ATR'] = atr_vals  # another measure of volatility

        #=======================================================================
        # 10. AVERAGE DIRECTIONAL INDEX (ADX) ----------------------------------
        #=======================================================================
        adx_vals, di_plus_vals, di_minus_vals = ti.get_adx_dis(
            X[[self.high_label, self.low_label, self.close_label]]
        )
        if self.normalize:
            X['ADX*'] = adx_vals / 100  # [0, 1]
            X['DI_HIST*'] = (di_plus_vals - di_minus_vals) / 100  # ~[-1, 1]
        else:
            X['ADX'] = adx_vals
            X['DI+'] = di_plus_vals
            X['DI-'] = di_minus_vals

        # Return the dataframe with all the features added
        return X


def main():
    return None


if __name__ == '__main__':
    main()
