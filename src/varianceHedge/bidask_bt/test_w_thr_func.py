import pandas as pd
import numpy as np
from src.varianceHedge.tools.op_w import compute_op_w
import logging
import sys
import gc

logging.basicConfig(filename='investigate/test_w_thr_func.log', level=logging.DEBUG)
# logging.basicConfig(filename='investigate/test_w_thr_func.log', level=logging.DEBUG)



pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

month_dt = '2021-04-01'


init_port_val = 10000


mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
bids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_bid_snapshot_5.pkl').ffill().resample('1min').last()
asks = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_ask_snapshot_5.pkl').ffill().resample('1min').last()

# syms = ['weighted_BTCGBP','weighted_ETHGBP']
syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

logging.info(f'bids')
logging.info(bids.head(61))

logging.info(f'asks')
logging.info(asks.head(61))

mids = mids[syms]
bids = bids[syms]
asks = asks[syms]



def f_opt_w(w_threshold):
    hist_mid = pd.DataFrame(columns=syms)
    hist_p = pd.DataFrame(columns=['MV', 'RP', 'MS', 'UW'])
    hist_w = {}
    hist_opt_w = {}
    hist_n = {}

    for dt, mid in mids.iterrows():

        ## Append Current Prices
        hist_mid.loc[dt] = mid
        for port in hist_p.columns:
            logging.info(port)
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


                logging.info(f'{port} init n: ')
                logging.info(hist_n[port])


                hist_opt_w[port] = hist_w[port].copy()
                hist_p.loc[dt, port] = init_port_val

                continue
            logging.info(f"dt: {dt}")
            curr_port_val = np.sum([hist_n[port].iloc[-1][symb] * bids.loc[dt][symb]
                                    if hist_n[port].iloc[-1][symb] > 0
                                    else hist_n[port].iloc[-1][symb] * asks.loc[dt][symb] for symb in syms])
            hist_p.loc[dt, port] = curr_port_val

            logging.info("hist_n[port].iloc[-1]")
            logging.info(hist_n[port].iloc[-1])
            logging.info("bids.loc[dt]")
            logging.info(bids.loc[dt])

            logging.info("asks.loc[dt]")
            logging.info(asks.loc[dt])

            logging.info("port_val ")
            logging.info([hist_n[port].iloc[-1][symb] * bids.loc[dt][symb]
                                    if hist_n[port].iloc[-1][symb] > 0
                                    else hist_n[port].iloc[-1][symb] * asks.loc[dt][symb] for symb in syms])
            logging.info('hist_p.loc[dt, port]')
            logging.info(hist_p.loc[dt, port])
            logging.info('hist_mid[-61:]')
            logging.info(hist_mid[-61:])

            logging.info('hist_w[port].iloc[-1]')
            logging.info(hist_w[port].iloc[-1])


            opt_w = pd.Series(getattr(compute_op_w,
                                      f"compute_{port}_weights")(
                cov=hist_mid[-61:].dropna(how='all', axis=1).pct_change().cov(),
                exp_ret=hist_mid[-61:].dropna(how='all', axis=1).pct_change().mean(),
                hist=hist_mid[-61:]),
                index=syms)

            hist_opt_w[port].loc[dt] = opt_w

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

        if dt == pd.to_datetime('2021-04-01 01:02:00'):
            break

    # [w.to_pickle(f'investigate/func_{w_name}.pkl') for w_name, w in hist_w.items()]
    # [w.to_pickle(f'investigate/func_opt_{w_name}.pkl') for w_name, w in hist_opt_w.items()]
    hist_p['w_threshold'] = w_threshold
    gc.collect()
    return hist_p


hist_df = f_opt_w(0.2)
print(hist_df)

