import pandas as pd
import numpy as np
import ray
pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


ray.init()
init_port_val = 10000
class compute_op_w:
    def compute_MV_weights(**kwargs):
        inv_covar = np.linalg.inv(kwargs['cov'])
        u = np.ones(len(kwargs['cov']))

        return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))

    def compute_MS_weights(**kwargs):
        inv_covar = np.linalg.inv(kwargs['cov'])
        u = np.ones(len(kwargs['cov']))
        return np.dot(inv_covar, kwargs['exp_ret']) / np.dot(u.T, np.dot(inv_covar, kwargs['exp_ret']))

    def compute_RP_weights(**kwargs):
        weights = (1 / np.diag(kwargs['cov']))
        return weights / sum(weights)

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]


mids = pd.read_pickle('datasets/consolidated_bitstamp_book_mid_snapshot_5.pkl').ffill().resample('1min').last()
bids = pd.read_pickle('datasets/consolidated_bitstamp_book_bid_snapshot_5.pkl').ffill().resample('1min').last()
asks = pd.read_pickle('datasets/consolidated_bitstamp_book_ask_snapshot_5.pkl').ffill().resample('1min').last()

# syms = ['weighted_BTCGBP','weighted_ETHGBP']
syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

print(syms)


mids = mids[syms]
bids = bids[syms]
asks = asks[syms]



@ray.remote
def f_opt_w(w_threshold):
    hist_mid = pd.DataFrame(columns=syms)
    hist_p = pd.DataFrame(columns=['MV', 'RP', 'MS', 'UW'])
    hist_w = {}
    hist_n = {}

    for dt, mid in mids.iterrows():

        for port in hist_p.columns:
            if dt < pd.to_datetime('2021-04-01 01:00:00'):
                hist_mid.loc[dt] = mid
                continue
            if dt == pd.to_datetime('2021-04-01 01:00:00'):
                hist_w[port] = pd.DataFrame(
                    [getattr(compute_op_w, f"compute_{port}_weights")(cov=hist_mid[-60:].pct_change().cov(),
                                                                      exp_ret=hist_mid[-60:].pct_change().mean())],
                    columns=syms,
                    index=[pd.to_datetime('2021-04-01 01:00:00')])
                hist_n[port] = (init_port_val * (hist_w[port].loc['2021-04-01 01:00:00']) / bids.loc[
                    '2021-04-01 01:00:00']).to_frame().T

            curr_port_val = np.sum(hist_n[port].iloc[-1] * bids.loc[dt])

            opt_w = pd.Series(getattr(compute_op_w,
                                      f"compute_{port}_weights")(cov=hist_mid[-60:].pct_change().cov(),
                                                                 exp_ret=hist_mid[-60:].pct_change().mean()),
                              index=syms)

            ## sort by diff, making always selling first before buying in rebalancing
            diff_w = (opt_w - hist_w[port].iloc[-1]).sort_values(ascending=True)

            prev_dt = hist_n[port].iloc[-1].name

            if diff_w.abs().max() > w_threshold:
                for sym, diff_w_i in diff_w.items():
                    ## buy at ask price, sell at bid price
                    n_i = diff_w_i * curr_port_val / asks.loc[dt, sym] if diff_w_i > 0 else diff_w_i * curr_port_val / \
                                                                                            bids.loc[dt, sym]
                    hist_n[port].loc[dt, sym] = hist_n[port].loc[prev_dt][sym] + n_i
            hist_w[port].loc[dt] = hist_n[port].iloc[-1] * bids.loc[dt] / np.sum(hist_n[port].iloc[-1] * bids.loc[dt])
            hist_p.loc[dt, port] = curr_port_val

    hist_p['w_threshold'] = w_threshold
    return hist_p


# 1bps change limit
futures = [f_opt_w.remote(i) for i in np.arange(0.0001, 0.125, 0.0001)]

dfs = ray.get(futures)
main_hist_p =pd.DataFrame()

for df in dfs:
    main_hist_p = main_hist_p.append(df)


main_hist_p.to_pickle('main_hist_p.pkl')