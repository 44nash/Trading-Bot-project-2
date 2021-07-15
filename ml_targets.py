import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


def get_frac_returns(timeseries: pd.Series, diff_period: int = 1) -> pd.Series:
    frac_returns = timeseries.pct_change(periods=diff_period)
    frac_returns[0:diff_period] = 0.0
    return frac_returns


def get_log_returns(timeseries: pd.Series, diff_period: int = 1) -> pd.Series:
    log_returns = np.log(timeseries / timeseries.shift(periods=diff_period))
    log_returns[0:diff_period] = 0.0
    return log_returns


def get_lookahead_returns(
    timeseries: pd.Series,
    lookahead_returns: List[int] = [1, 2, 3, 4, 5],
    normalize: bool = False,
    output_label_base: str = 'LookaheadReturn',
    calc_method: str = 'Log',
) -> pd.DataFrame:
    """
    """
    # How should the returns be calculated?
    if calc_method.lower() == 'log':
        returns_fcn = get_log_returns
    else:
        get_frac_returns

    returns_dict = {}
    for ell in lookahead_returns:
        # Construct output column name
        colname = f"{output_label_base}_{ell}"
        # Calculate future returns.  NB: We are calculating future returns,
        # then rolling back the returns to the current timebar.
        returns_dict[colname] = returns_fcn(
            timeseries,
            diff_period=ell  # roll forward
        ).shift(periods=-ell)  # roll backward

    # Convert returns to a `pandas.DataFrame`
    df = pd.DataFrame(data=returns_dict, index=timeseries.index)

    # Normalize returns by subtracting the mean and dividing by the standard deviation
    if normalize:
        df = (df - df.mean(axis=0)) / df.std(axis=0)

    # Return the result
    return df


class MLTargetsAdder(BaseEstimator, TransformerMixin):
    """
    """

    def __init__(
        self,
        lookahead_returns: List[int] = [1, 2, 3, 4, 5],
        output_label_base: str = 'LookaheadReturns',
        normalize: bool = False,
        input_label: str = 'Close',
        calc_method: str = 'Log',
    ) -> None:
        super().__init__()
        self.lookahead_returns = lookahead_returns
        self.output_label_base = output_label_base
        self.normalize = normalize
        self.input_label = input_label
        self.calc_method = calc_method
        return None

    def fit(self, X: pd.DataFrame, **fit_params):
        return self

    def transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        return X.join(
            get_lookahead_returns(
                X[self.input_label],
                lookahead_returns=self.lookahead_returns,
                normalize=self.normalize,
                output_label_base=self.output_label_base,
                calc_method=self.calc_method
            )
        )


def main() -> None:
    return None


if __name__ == '__main__':
    main()
