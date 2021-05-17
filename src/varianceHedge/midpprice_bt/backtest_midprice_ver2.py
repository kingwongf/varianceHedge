import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from src.varianceHedge.tools.op_w import compute_op_w
from tqdm import tqdm

pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


init_port_val = 10000

w_threshold = 0.20
price_sampling_size = 61



mids = pd.read_pickle('datasets/consolidated_binance.pkl').astype('float64').ffill().resample('1min').last()

syms = ['btcgbp', 'ethgbp', 'xrpgbp'] # mids.columns

mids = mids[syms]





# st_bt = mids.index[0] + pd.Timedelta(f"{price_sampling_size}M")
st_bt = pd.to_datetime('2020-09-01 00:00:00')

hist_mid = pd.DataFrame(columns=syms)
hist_p = pd.DataFrame(columns=['MV','NRP', 'RP', 'MS', 'UW'])
hist_w = {}
hist_n = {}
hist_opt_w ={}


for dt, mid in tqdm(mids.iterrows(), total=mids.shape[0]):
    # print(f"current dt: {dt}")
    ## Append Current Prices
    hist_mid.loc[dt] = mid
    for port in hist_p.columns:
        if dt <pd.to_datetime(st_bt):

            continue
        if dt==pd.to_datetime(st_bt):
            hist_w[port] = pd.DataFrame([getattr(compute_op_w, f"compute_{port}_weights")(
                cov=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().cov(),
                exp_ret=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().mean())],
                columns=syms,
                index=[st_bt])

            hist_n[port] = (init_port_val * (hist_w[port].loc[st_bt]) / mids.loc[st_bt]).to_frame().T

            hist_opt_w[port] = pd.DataFrame([getattr(compute_op_w, f"compute_{port}_weights")(
                cov=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().cov(),
                exp_ret=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().mean())],
                columns=syms,
                index=[st_bt])


        ## Previous n holdings multiply current prices
        curr_port_val = np.sum(hist_n[port].iloc[-1]*mids.loc[dt])
        hist_p.loc[dt, port] = curr_port_val

        # print('port')
        # print(port)
        # print('syms')
        # print(syms)
        # print('cov')
        # print(hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().cov())
        # print('w')
        # print(getattr(compute_op_w,
        #                           f"compute_{port}_weights")(cov=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().cov(),
        #                                                      exp_ret=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().mean()))
        #

        opt_w = pd.Series(getattr(compute_op_w,
                                  f"compute_{port}_weights")(cov=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().cov(),
                                                             exp_ret=hist_mid[-price_sampling_size:].dropna(how='all', axis=1).pct_change().mean()),
                          index=syms)

        hist_opt_w[port].loc[dt] = opt_w


        ## sort by diff, making always selling first before buying in rebalancing
        diff_w = (opt_w - hist_w[port].iloc[-1]).sort_values(ascending=True)
        prev_dt = hist_n[port].iloc[-1].name

        # print(f"{port} prev dt: {prev_dt}")
        ## TODO changed from mean to max
        if diff_w.abs().max() > w_threshold:
            for sym, diff_w_i in diff_w.items():
                n_i = diff_w_i*curr_port_val / mids.loc[dt, sym]
                hist_n[port].loc[dt, sym] = hist_n[port].loc[prev_dt][sym] + n_i



        # print(hist_n[port])
        hist_w[port].loc[dt] = hist_n[port].iloc[-1] * mids.loc[dt] / np.sum(hist_n[port].iloc[-1] * mids.loc[dt])


hist_p.to_pickle(f"results/binance/bt_mid_sampling_{price_sampling_size}M.pkl")

fig = hist_p.plot()
fig.update_yaxes(title_text='GBP')
fig.write_html(f"results/binance/bt_mid_sampling_{price_sampling_size}M.html")
fig.write_image(f"results/binance/bt_mid_sampling_{price_sampling_size}M.png")


    # hist_w_MV = hist_w['MV']
    # hist_opt_w_MV = hist_opt_w['MV']
    #
    # print('hist_w_MV')
    # print(hist_w_MV)
    #
    # print('hist_opt_w_MV')
    # hist_opt_w_MV = hist_opt_w_MV.shift(1).fillna(0)
    # print(hist_opt_w_MV)
    #
    # hist_opt_w_MV.loc['2021-04-01 01:00:00'] = 0
    # hist_opt_port = ((hist_opt_w_MV * mids.pct_change()).loc['2021-04-01 01:00:00':].fillna(0).sum(
    #     axis=1) + 1).cumprod() * init_port_val
    #
    # print('apply_w_MV')
    # hist_cov = mids.pct_change().rolling('60min', min_periods=60).cov().dropna()
    # print('apply_cov')
    # print(hist_cov)
    # apply_w_MV = hist_cov.groupby(level=0, axis=0).apply(apply_compute_MV_weights).apply(pd.Series).dropna().shift(1)
    # apply_w_MV.columns = mids.columns
    # print(apply_w_MV)
    #
    # hist_w_MV = hist_w_MV.shift(1).fillna(0)
    # mv_port = ((hist_w_MV * mids.pct_change()).loc['2021-04-01 01:00:00':].fillna(0).sum(axis=1) + 1).cumprod() * init_port_val
    #
    # apply_w_MV.loc['2021-04-01 01:00:00'] = 0
    # apply_mv_port = ((apply_w_MV * mids.pct_change()).loc['2021-04-01 01:00:00':].fillna(0).sum(axis=1) + 1).cumprod() * init_port_val
    # print('apply_mv_port')
    # print(apply_mv_port)
    #
    # compare_df = pd.concat([mv_port, hist_p['MV'], apply_mv_port.rename('apply_w_MV'), hist_opt_port.rename('hist_opt')], axis=1).rename(columns={0:'hist_w_MV'}).astype('float64')
    # print('compare_df')
    # print(compare_df)
    #fig = compare_df.plot()
    #fig.show()


    # print(month_dt)
    # print(hist_p)