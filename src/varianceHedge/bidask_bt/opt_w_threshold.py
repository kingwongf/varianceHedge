import pandas as pd
import numpy as np
from pypfopt.cla import CLA
import ray
import gc
pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

month_dt = '2021-04-01'


init_port_val = 10000
class compute_op_w:


    def compute_MV_weights(**kwargs):
        try:
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))
        except Exception as e:
            kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            w = np.append(np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u)), 0)
            return w

    # def compute_MS_weights(**kwargs):
    #     if not compute_op_w._is_pos_def(kwargs['cov']):
    #         kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
    #         kwargs['exp_ret'] = kwargs['exp_ret'].drop('weighted_OMGGBP')
    #
    #     inv_covar = np.linalg.inv(kwargs['cov'])
    #     u = np.ones(len(kwargs['cov']))
    #     w = np.dot(inv_covar, kwargs['exp_ret']) / np.dot(u.T, np.dot(inv_covar, kwargs['exp_ret']))
    #
    #     w = w if w.shape[0]==6 else np.append(w, 0)
    #    return w
    @staticmethod
    def _is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def compute_RP_weights(**kwargs):
        weights = (1 / np.diag(kwargs['cov']))
        return weights / sum(weights)

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]

    def compute_MS_weights(**kwargs):
        try:
            w = pd.Series(CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()).loc[kwargs['exp_ret'].index].values
            return w
        except Exception as e:
            kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
            kwargs['exp_ret'] = kwargs['exp_ret'].drop('weighted_OMGGBP')

            w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()
            w.update({'weighted_OMGGBP':0})

            return pd.Series(w).loc[kwargs['exp_ret'].index.append(pd.Index(['weighted_OMGGBP']))].values

mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
bids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
asks = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()

# syms = ['weighted_BTCGBP','weighted_ETHGBP']
syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

print(syms)


mids = mids[syms]
bids = bids[syms]
asks = asks[syms]



def f_opt_w(w_threshold):
    hist_mid = pd.DataFrame(columns=syms)
    hist_p = pd.DataFrame(columns=['MV', 'RP', 'MS', 'UW'])
    hist_w = {}
    hist_n = {}

    for dt, mid in mids.iterrows():
        ## Append Current Prices
        hist_mid.loc[dt] = mid
        for port in hist_p.columns:
            if dt<pd.to_datetime(f'{month_dt} 01:00:00'):
                continue

            if dt==pd.to_datetime(f'{month_dt} 01:00:00'):
                hist_w[port] = pd.DataFrame([getattr(compute_op_w, f"compute_{port}_weights")(
                    cov=hist_mid[-61:].dropna(how='all', axis=1).pct_change().cov(),
                    exp_ret=hist_mid[-61:].dropna(how='all', axis=1).pct_change().mean())],
                    columns=syms,
                    index=[pd.to_datetime(f'{month_dt} 01:00:00')])
                hist_n[port] = (init_port_val * (hist_w[port].loc[f'{month_dt} 01:00:00']) / bids.loc[
                    f'{month_dt} 01:00:00']).to_frame().T

                hist_p.loc[dt, port] = init_port_val

                continue

            curr_port_val = np.sum([hist_n[port].iloc[-1][symb] * bids.loc[dt][symb]
                                    if hist_n[port].iloc[-1][symb] > 0
                                    else hist_n[port].iloc[-1][symb] * asks.loc[dt][symb] for symb in syms])
            hist_p.loc[dt, port] = curr_port_val

            opt_w = pd.Series(getattr(compute_op_w,
                                      f"compute_{port}_weights")(
                cov=hist_mid[-61:].dropna(how='all', axis=1).pct_change().cov(),
                exp_ret=hist_mid[-61:].dropna(how='all', axis=1).pct_change().mean(),
                hist=hist_mid[-61:]),
                index=syms)

            ## sort by diff, making always selling first before buying in rebalancing
            diff_w = (opt_w - hist_w[port].iloc[-1]).sort_values(ascending=True)
            prev_dt = hist_n[port].iloc[-1].name

            ## TODO changed from mean to max
            if diff_w.abs().max() > w_threshold:

                for sym, diff_w_i in diff_w.items():
                    ## buy at ask price, sell at bid price
                    n_i = diff_w_i * curr_port_val / asks.loc[dt, sym] \
                        if diff_w_i > 0 else diff_w_i * curr_port_val / bids.loc[dt, sym]
                    hist_n[port].loc[dt, sym] = hist_n[port].loc[prev_dt][sym] + n_i

            hist_w[port].loc[dt] = hist_n[port].iloc[-1] * bids.loc[dt] / np.sum(hist_n[port].iloc[-1] * bids.loc[dt])

    hist_p['w_threshold'] = w_threshold
    gc.collect()
    return hist_p


# 100 bps change limit
dfs = [f_opt_w(i) for i in np.arange(0.00, 0.51, 0.01)]

main_hist_p =pd.DataFrame()

for df in dfs:
    main_hist_p = main_hist_p.append(df)


main_hist_p.to_pickle(f'main_hist_unserialise_{month_dt}_p.pkl')