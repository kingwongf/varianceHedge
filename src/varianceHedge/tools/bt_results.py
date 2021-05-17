import pandas as pd
import numpy as np
from scipy.stats import skew
import swifter


def dd(ts):
    return np.min(ts / np.maximum.accumulate(ts)) - 1

def rolling_maxdd(equity_curve):
    return equity_curve.rolling(262800).apply(dd).dropna()

def maxdd(equity_curve):
    return rolling_maxdd(equity_curve).min()


def np_maxdd(equity_curve):
    return np.min(equity_curve -  np.maximum.accumulate(equity_curve))

def _np_stats(np_equity_curves: np.array):



    ret = np.diff(np_equity_curves, axis=0) / np_equity_curves[1:, :]
    prnt('np rets')
    print(ret)

    print(np.mean(ret, axis=0))
    np.std(ret, axis=0)

    skew(ret, axis=0)
    np.mean(ret, axis=0)/ np.std(ret, axis=0)


def _stats(equity_curve):
	return pd.Series([
			(equity_curve.iloc[-1] - 10000 )/ 10000,
	        equity_curve.pct_change().mean(),
	        equity_curve.pct_change().std(),
	        equity_curve.pct_change().skew(),
	        equity_curve.pct_change().mean()/ equity_curve.pct_change().std()]) #maxdd(equity_curve)


def stats(df_equity_curves):
    # df_equity_curves.values
    # r = df_equity_curves.swifter.apply(_stats)
    # r = df_equity_curves.apply(_stats)
    # r = pd.concat([_stats(df_equity_curves[port]) for port in df_equity_curves])

    # print(df_equity_curves)

    _np_stats(df_equity_curves.values)

    print('pandas ret')
    print(df_equity_curves.pct_change())

    r = pd.concat([_stats(df_equity_curves[port]) for port in df_equity_curves], axis=1)


    r.index = ['avg. daily Cumulative Returns',
               'avg. 1min Returns',
               'avg. 1min Vol',
               'avg. 1min Skew',
               'avg. 1min Sharpe',] #'avg. daily MaxDD'
    r.columns = df_equity_curves.columns

    print(r)