import pandas as pd
import numpy as np
dts = pd.date_range(start='11/1/2020', end='04/1/2021', freq='MS').strftime("%Y-%m-%d")
port_dfs = [pd.read_pickle(f"results/bt_{month_dt}.pkl") for month_dt in dts]



def dd(ts):
    return np.min(ts / np.maximum.accumulate(ts)) - 1

def rolling_maxdd(equity_curve):
    return equity_curve.rolling(126).apply(dd).dropna()

def maxdd(equity_curve):
    return rolling_maxdd(equity_curve).min()

def stats(equity_curve):
	return pd.DataFrame([
			(equity_curve.iloc[-1].rename(None) - 10000 )/ 10000,
	        equity_curve.pct_change().mean(),
	        equity_curve.pct_change().std(),
	        equity_curve.pct_change().skew(),
	        equity_curve.pct_change().mean()/ equity_curve.pct_change().std(),
	        maxdd(equity_curve)])

r = pd.concat([stats(port) for port in port_dfs ])
r = r.groupby(r.index).mean()
r.index = ['avg. daily Cumulative Returns',
           'avg. 1min Returns',
           'avg. 1min Vol',
           'avg. 1min Skew',
           'avg. 1min Sharpe',
           'avg. daily MaxDD',]

r.to_csv('r.csv')
print(r)