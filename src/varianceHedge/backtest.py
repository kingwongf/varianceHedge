import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.cla import CLA


pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


init_port_val = 10000

w_threshold = 0.01

dts = pd.date_range(start='11/1/2020', end='04/1/2021', freq='MS').strftime("%Y-%m-%d")

class compute_op_w:
    def compute_MV_weights(**kwargs):
        inv_covar = np.linalg.inv(kwargs['cov'])
        u = np.ones(len(kwargs['cov']))

        return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))

    # def compute_MS_weights(**kwargs):
    #     inv_covar = np.linalg.inv(kwargs['cov'])
    #     u = np.ones(len(kwargs['cov']))
    #     return np.dot(inv_covar, kwargs['exp_ret']) / np.dot(u.T, np.dot(inv_covar, kwargs['exp_ret']))

    def _is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def compute_RP_weights(**kwargs):
        weights = (1 / np.diag(kwargs['cov']))
        return weights / sum(weights)

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]

    def compute_MS_weights(**kwargs):
        # print(kwargs['exp_ret'])
        # print(kwargs['cov'])
        # print(compute_op_w._is_pos_def(kwargs['cov']))
        # return pd.Series(EfficientFrontier(kwargs['exp_ret'], kwargs['cov'], solver='ECOS').max_sharpe(risk_free_rate=0.0)).loc[kwargs['exp_ret'].index].values
        return pd.Series(CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(0, 1)).max_sharpe()).loc[kwargs['exp_ret'].index].values



for month_dt in dts:

    mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
    bids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_bid_snapshot_5.pkl').ffill().resample('1min').last()
    asks = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_ask_snapshot_5.pkl').ffill().resample('1min').last()

    # syms = ['weighted_BTCGBP','weighted_ETHGBP']
    syms = mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')].drop(pd.Index(['weighted_OMGGBP'])) if month_dt in ['2020-11-01', '2020-12-01', '2021-01-01'] else mids.columns[mids.columns.str.match(r'(^weighted.*GBP$)')]

    print(month_dt)
    print(syms)
    # syms = mids.columns[mids.columns.str.match(r'(^best.*GBP$)')]



    mids = mids[syms]
    bids = bids[syms]
    asks = asks[syms]



    # hist_mid = mids.loc[pd.to_datetime('2021-04-01 01:00:00')].to_frame().T
    hist_mid = pd.DataFrame(columns=syms)
    hist_p = pd.DataFrame(columns=['MV', 'RP', 'MS', 'UW'])
    hist_w = {}
    hist_n = {}


    for dt, mid in mids.iterrows():

        for port in hist_p.columns:
            if dt <pd.to_datetime(f'{month_dt} 01:00:00'):
                hist_mid.loc[dt] = mid
                continue
            if dt==pd.to_datetime(f'{month_dt} 01:00:00'):
                hist_w[port] = pd.DataFrame([getattr(compute_op_w, f"compute_{port}_weights")(cov=hist_mid[-60:].dropna(how='all', axis=1).pct_change().cov(),
                                                                              exp_ret=hist_mid[-60:].dropna(how='all', axis=1).pct_change().mean())],
                                      columns=syms,
                                      index=[pd.to_datetime(f'{month_dt} 01:00:00')])
                hist_n[port] = (init_port_val * (hist_w[port].loc[f'{month_dt} 01:00:00']) / bids.loc[f'{month_dt} 01:00:00']).to_frame().T


            curr_port_val = np.sum([hist_n[port].iloc[-1][symb]*bids.loc[dt][symb] if hist_n[port].iloc[-1][symb] > 0 else hist_n[port].iloc[-1][symb]*asks.loc[dt][symb] for symb in syms])


            opt_w = pd.Series(getattr(compute_op_w,
                                      f"compute_{port}_weights")(cov=hist_mid[-60:].dropna(how='all', axis=1).pct_change().cov(),
                                                                 exp_ret=hist_mid[-60:].dropna(how='all', axis=1).pct_change().mean()),
                              index=syms)


            ## sort by diff, making always selling first before buying in rebalancing
            diff_w = (opt_w - hist_w[port].iloc[-1]).sort_values(ascending=True)

            # if port=='MS' and month_dt=='2021-01-01' and pd.to_datetime("2021-01-01 06:12:00")>=dt>=pd.to_datetime("2021-01-01 06:06:00"):
            #     print(dt)
            #     print(f"curr_port_val: {curr_port_val}")
            #     print('opt_w')
            #     print(opt_w)
            #
            #     print('diff_w')
            #     print(diff_w)
            #     print('curr_w')
            #     print(hist_w[port].iloc[-1])
            #



            prev_dt = hist_n[port].iloc[-1].name

            #print(diff_w)
            #print(diff_w.abs().mean())

            ## TODO changed from mean to max
            if diff_w.abs().max() > w_threshold:
                for sym, diff_w_i in diff_w.items():
                    ## buy at ask price, sell at bid price
                    n_i = diff_w_i*curr_port_val / asks.loc[dt, sym] if diff_w_i > 0 else diff_w_i*curr_port_val / bids.loc[dt,sym]
                    hist_n[port].loc[dt, sym] = hist_n[port].loc[prev_dt][sym] + n_i

            # if dt==pd.to_datetime('2021-04-01 08:19:00') and port=='UW':
            #     print(hist_w[port])
            #     print(opt_w)
            #     print(diff_w)
            # if dt==pd.to_datetime('2021-04-01 08:20:00') and port=='UW':
            #     print(hist_w[port])
            #     print(opt_w)
            #     print(diff_w)
            # if dt==pd.to_datetime('2021-04-01 08:21:00') and port=='UW':
            #     print(hist_w[port])
            #     print(opt_w)
            #     print(diff_w)


            # print(hist_n[port])
            hist_w[port].loc[dt] = hist_n[port].iloc[-1] * bids.loc[dt] / np.sum(hist_n[port].iloc[-1] * bids.loc[dt])
            hist_p.loc[dt, port] = curr_port_val



    # for index, df in hist_w.items():

        # print(index+"_weight")
        #
        # print(df)
        # print(index + "_n")
        # print(hist_n[index])



    fig = hist_p.plot()
    fig.update_yaxes(title_text='GBP')
    fig.write_html(f"results/bt_{month_dt}.html")
    fig.write_image(f"results/bt_{month_dt}.png")
    hist_p.to_pickle(f"results/bt_{month_dt}.pkl")
    # print(month_dt)
    # print(hist_p)