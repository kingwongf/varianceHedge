import pandas as pd
import numpy as np
from src.varianceHedge.tools.op_w import compute_op_w
import ray
pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

month_dt = '2021-04-01'
ray.init()
init_port_val = 10000


mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()


# syms = ['weighted_BTCGBP','weighted_ETHGBP']
syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

print(syms)


mids = mids[syms]



@ray.remote
def f_opt_w(w_threshold):
    hist_mid = pd.DataFrame(columns=syms)
    hist_p = pd.DataFrame(columns=['MV', 'RP', 'MS', 'UW'])
    hist_w = {}
    hist_n = {}

    for dt, mid in mids.iterrows():
        ## Append Current Prices
        hist_mid.loc[dt] = mid
        for port in hist_p.columns:
            if dt < pd.to_datetime(f'{month_dt} 01:00:00'):
                continue

            if dt == pd.to_datetime(f'{month_dt} 01:00:00'):
                hist_w[port] = pd.DataFrame([getattr(compute_op_w, f"compute_{port}_weights")(
                    cov=hist_mid[-61:].dropna(how='all', axis=1).pct_change().cov(),
                    exp_ret=hist_mid[-61:].dropna(how='all', axis=1).pct_change().mean())],
                                            columns=syms,
                                            index=[pd.to_datetime(f'{month_dt} 01:00:00')])

                hist_n[port] = (init_port_val * (hist_w[port].loc[f'{month_dt} 01:00:00']) / mids.loc[f'{month_dt} 01:00:00']).to_frame().T
                hist_p.loc[dt, port] = init_port_val

                continue

            curr_port_val = np.sum(hist_n[port].iloc[-1] * mids.loc[dt])
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
                    n_i = diff_w_i*curr_port_val / mids.loc[dt, sym]
                    hist_n[port].loc[dt, sym] = hist_n[port].loc[prev_dt][sym] + n_i

            hist_w[port].loc[dt] = hist_n[port].iloc[-1] * mids.loc[dt] / np.sum(hist_n[port].iloc[-1] * mids.loc[dt])

    hist_p['w_threshold'] = w_threshold
    return hist_p


# 100 bps change limit
futures = [f_opt_w.remote(i) for i in np.arange(0.00, 0.101, 0.01)]

dfs = ray.get(futures)
main_hist_p =pd.DataFrame()

for df in dfs:
    main_hist_p = main_hist_p.append(df)


main_hist_p.to_pickle(f'main_hist_mid_{month_dt}_p.pkl')