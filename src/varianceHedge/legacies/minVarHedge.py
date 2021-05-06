import pandas as pd
import numpy as np
pd.options.plotting.backend = "plotly"
init_port_val = 10000
def compute_MV_weights(covariances):
    inv_covar = np.linalg.inv(covariances)
    u = np.ones(len(covariances))

    return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))

def compute_MS_weights(exp_rets, covariances):
    inv_covar = np.linalg.inv(covariances)
    u = np.ones(len(covariances))
    return np.dot(inv_covar, exp_rets) / np.dot(u.T, np.dot(inv_covar, exp_rets))

def compute_RP_weights(covariances):
    weights = (1 / np.diag(covariances))
    return weights / sum(weights)


mids = pd.read_pickle('datasets/consolidated_bitstamp_book_mid_snapshot_5.pkl').ffill().resample('1min').last()
bids = pd.read_pickle('datasets/consolidated_bitstamp_book_bid_snapshot_5.pkl').ffill().resample('1min').last()
asks = pd.read_pickle('datasets/consolidated_bitstamp_book_ask_snapshot_5.pkl').ffill().resample('1min').last()

# syms = ['weighted_BTCGBP','weighted_ETHGBP']
syms = mids.columns[mids.columns.str.startswith('weighted_')]



mids = mids[syms]
bids = bids[syms]
asks = asks[syms]


w_threshold = 0.005
# hist_mid = mids.loc[pd.to_datetime('2021-04-01 01:00:00')].to_frame().T
hist_mid = pd.DataFrame(columns=syms)
hist_p = pd.Series()


for dt, mid in pd.read_pickle('datasets/consolidated_bitstamp_book_mid_snapshot_5.pkl').ffill().resample('1min').last().iterrows():
    if dt <pd.to_datetime('2021-04-01 01:00:00'):
        hist_mid.loc[dt] = mid
        continue
    if dt==pd.to_datetime('2021-04-01 01:00:00'):
        hist_w = pd.DataFrame([compute_MV_weights(hist_mid.pct_change().cov())],
                              columns=syms,
                              index=[pd.to_datetime('2021-04-01 01:00:00')])
        hist_n = (init_port_val * (hist_w.loc['2021-04-01 01:00:00']) / bids.loc['2021-04-01 01:00:00']).to_frame().T
    # print('hist_mid')
    # print('hist_n.iloc[-1]')
    # print(hist_n.iloc[-1])
    # print('bids.loc[dt]')
    # print(bids.loc[dt])
    curr_port_val = np.sum(hist_n.iloc[-1]*bids.loc[dt])

    # print(f"curr_port_val: {curr_port_val}")
    opt_w = pd.Series(compute_MV_weights(hist_mid[-60:].pct_change().cov()), index=syms)

    # print(f'curr_port_val: {curr_port_val}')
    # print(dt)
    # print('opt_w')
    # print(opt_w)
    # print('curr_w')
    # print(hist_w.iloc[-1])


    #
    # print('hist_w.iloc[-1]')
    # print(hist_w.iloc[-1])

    diff_w = (opt_w - hist_w.iloc[-1]).sort_values(ascending=True)
    if (opt_w < 0).any():
        print('opt_w')
        print(opt_w)
        print(hist_n.iloc[-1])



        #
    # print(hist_w.iloc[-1])
    # print(diff_w)
    # print(dt)
    # print('hist_n')
    # print(hist_n)

    prev_dt = hist_n.iloc[-1].name
    for sym, diff_w_i in diff_w.items():
        if abs(diff_w_i) > w_threshold:
            # print(sym, diff_w_i)
            n_i = diff_w_i*curr_port_val / asks.loc[dt, sym] if diff_w_i > 0 else diff_w_i*curr_port_val / bids.loc[dt,sym]
            # print('n_i')
            # print(n_i)
            #
            # print('hist_n before')
            # print(hist_n)
            # #
            # print('hist_n.loc[dt, sym] ')
            # print(hist_n.loc[dt, sym] )
            hist_n.loc[dt, sym] = hist_n.loc[prev_dt][sym] + n_i

            # print('hist_n after')
            # print(hist_n)

            # hist_w.loc[dt, sym] = hist_n.loc[dt, sym] * (bids.loc[dt,sym] + asks.loc[dt, sym])/2

    hist_w.loc[dt] = hist_n.iloc[-1] * mids.loc[dt] / np.sum(hist_n.iloc[-1] * mids.loc[dt])
    hist_p.loc[dt] = curr_port_val



fig = hist_p.plot()
fig.show()