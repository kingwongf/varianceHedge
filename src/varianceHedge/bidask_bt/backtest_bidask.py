import pandas as pd
import numpy as np
import logging
import gc
import sys
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA

logging.basicConfig(filename='investigate/backtest.log', level=logging.DEBUG)


pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


init_port_val = 10000

w_threshold = 0.20

# dts = pd.date_range(start='11/1/2020', end='04/1/2021', freq='MS').strftime("%Y-%m-%d")
dts = ['2021-04-01']
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


for month_dt in dts:

    mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
    bids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_bid_snapshot_5.pkl').ffill().resample('1min').last()
    asks = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_ask_snapshot_5.pkl').ffill().resample('1min').last()

    logging.info(f'bids')
    logging.info(bids.head(61))

    logging.info(f'asks')
    logging.info(asks.head(61))


    syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

    # syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')].drop(pd.Index(['weighted_OMGGBP'])) \
    #     if month_dt in ['2020-11-01', '2020-12-01', '2021-01-01'] \
    #     else mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]



    mids = mids[syms]
    bids = bids[syms]
    asks = asks[syms]



    # hist_mid = mids.loc[pd.to_datetime('2021-04-01 01:00:00')].to_frame().T
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
            gc.collect()

        if dt==pd.to_datetime('2021-04-01 01:02:00'):
            break

    # [w.to_pickle(f'investigate/nonfunc_{w_name}.pkl') for w_name, w in hist_w.items()]
    # [w.to_pickle(f'investigate/nonfunc_opt_{w_name}.pkl') for w_name, w in hist_opt_w.items()]

    logging.info('hist_p')
    logging.info(hist_p)

    # print('hist_n[MV]')
    # print(hist_n['MV'])
    #
    # print(hist_w['MV'] )
    # print((hist_w['MV'] * mids['2021-04-01 01:00:00':].pct_change()).fillna(0))
    # print((((hist_w['MV'] * mids['2021-04-01 01:00:00':].pct_change()).fillna(0)).sum(axis=1) + 1).cumprod())
    # fig = (((hist_w['MV'] * mids['2021-04-01 01:00:00':].pct_change()).fillna(0)).sum(axis=1) + 1).cumprod().plot()
    # fig.show()



    # fig = hist_p.plot()
    # fig.update_yaxes(title_text='GBP')
    # fig.write_html(f"results/bt_bid_ask_{month_dt}.html")
    # fig.write_image(f"results/bt_bid_ask_{month_dt}.png")
    # hist_p.to_pickle(f"results/bt_bid_ask_{month_dt}.pkl")
    # print(month_dt)
    # print(hist_p)