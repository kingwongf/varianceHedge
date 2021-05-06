import pandas as pd
import numpy as np
from pypfopt.cla import CLA

import seaborn as sns
pd.options.plotting.backend = "plotly"
from scipy import stats
import matplotlib.pyplot as plt


class compute_op_w:
    def compute_MV_weights(**kwargs):
        inv_covar = np.linalg.inv(kwargs['cov'])
        u = np.ones(len(kwargs['cov']))

        return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))

    # def compute_MS_weights(**kwargs):
    #     inv_covar = np.linalg.inv(kwargs['cov'])
    #     u = np.ones(len(kwargs['cov']))
    #     return np.dot(inv_covar, kwargs['exp_ret']) / np.dot(u.T, np.dot(inv_covar, kwargs['exp_ret']))
    @staticmethod
    def _is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    def compute_RP_weights(**kwargs):
        weights = (1 / np.diag(kwargs['cov']))
        return weights / sum(weights)

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]

    def compute_MS_weights(**kwargs):
        return pd.Series(CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(0, 1)).max_sharpe()).loc[kwargs['exp_ret'].index].values




df = pd.read_pickle('datasets/consolidated_bitstamp_book_2021-04-01_mid_snapshot_5.pkl').ffill().resample('1min').last()
df = df[df.columns[df.columns.str.startswith('weighted_')]]
rets = df.pct_change().dropna()

sym = rets.columns

## min. variance hedge
## variances-covariances sampling period 1hr

rolling_60_mins_cov_df = rets.rolling('60min', min_periods=60).cov().dropna(axis=0)


# rolling_cov_dfs = [rolling_60_mins_cov_df, rolling_120_mins_cov_df,
#                    rolling_resample_5_mins_cov_df, rolling_resample_10_mins_cov_df,
#                    rolling_strict_resample_5_mins_cov_df, rolling_strict_resample_10_mins_cov_df]


rolling_cov_dfs = [rolling_60_mins_cov_df]
# cov_names = ['rolling 60mins', 'rolling_120mins', 'resample_5mins', 'resample_10mins', 'strict_5mins,', 'strict_10mins']
cov_names = ['rolling 60mins']

names_w = ['w_MV', 'w_RP', 'w_UW', 'w_MS']
## dynamic daily minimum weights


port = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_product([names_w, cov_names]))

bt_dt_range = pd.date_range(start='2021-04-01 01:00:00', end='2021-04-01 23:59:00', freq='1min')
for cov, cov_name in zip(rolling_cov_dfs, cov_names):





    w_MS = pd.DataFrame(index=bt_dt_range, columns= rets.columns)
    w_UW = pd.DataFrame(index=bt_dt_range, columns= rets.columns)
    w_MV = pd.DataFrame(index=bt_dt_range, columns= rets.columns)
    w_RP = pd.DataFrame(index=bt_dt_range, columns= rets.columns)


    for dt, _ in rets.iterrows():
        if dt < pd.to_datetime(f'2021-04-01 01:00:00'):
            continue


        # w_MS.loc[dt] = compute_op_w.compute_MS_weights(exp_ret=exp_rets.loc[dt], cov=cov.loc[dt])
        # w_UW.loc[dt] = compute_op_w.compute_UW_weights(exp_ret=exp_rets.loc[dt], cov=cov.loc[dt])
        # w_MV.loc[dt] = compute_op_w.compute_MV_weights(exp_ret=exp_rets.loc[dt], cov=cov.loc[dt])
        # w_RP.loc[dt] = compute_op_w.compute_RP_weights(exp_ret=exp_rets.loc[dt], cov=cov.loc[dt])

        cur_index = df.index.get_loc(dt)
        w_MS.loc[dt] = compute_op_w.compute_MS_weights(exp_ret=df.iloc[cur_index-60: cur_index].pct_change().mean(),
                                                       cov=df.iloc[cur_index-60: cur_index].pct_change().cov())
        w_UW.loc[dt] = compute_op_w.compute_UW_weights(exp_ret=df.iloc[cur_index-60: cur_index].pct_change().mean(),
                                                       cov=df.iloc[cur_index-60: cur_index].pct_change().cov())
        w_MV.loc[dt] = compute_op_w.compute_MV_weights(exp_ret=df.iloc[cur_index-60: cur_index].pct_change().mean(),
                                                       cov=df.iloc[cur_index-60: cur_index].pct_change().cov())
        w_RP.loc[dt] = compute_op_w.compute_RP_weights(exp_ret=df.iloc[cur_index-60: cur_index].pct_change().mean(),
                                                       cov=df.iloc[cur_index-60: cur_index].pct_change().cov())



    for w, name_w in zip([w_MV, w_RP, w_UW, w_MS], names_w):


        w = w.shift(1)
        print(name_w)
        print(w)
        print(w.sum(axis=1))

        print(rets[sym])

        ret_p = (w*rets[sym].loc[w.index]).sum(axis=1)

        print(ret_p)
        ret_p.iloc[0] = 0

        port[(name_w, cov_name)] = (ret_p + 1).cumprod()

        # vol = '{:f}'.format(ret_p.std())
        # vol_df.loc[name_w, cov_name] = ret_p.std()
        # print(f"{name_w} volatility: {vol}")

    # break

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df)
    # print(vol_df)
    # port[['w_MV', 'w_RP', 'w_UW']].plot()
    # plt.show()

# print(port)
idx = pd.IndexSlice
# port.columns.get_level_values(0)
# fig = port.loc[:, (port.columns.get_level_values(0)=='w_MV' | port.columns.get_level_values(0)=='w_RP') ].plot()
plot_port = port[names_w] * 10000
plot_port.columns = [' '.join(col).strip() for col in plot_port.columns.values]

# print(plot_port)
fig = plot_port.plot()
fig.show()
# plt.show()
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#
#     print(w_MV.sum(axis=1))
#     print(w_RP)
# # z_threshold = 2.58
# z = np.abs(stats.zscore(df, nan_policy='omit'))
#
# print(df.shape)
# df = df[(z <= z_threshold).any(axis=1)]
# print(df.shape)
#
# sns.pairplot(df[df.columns[df.columns.str.startswith('weighted_')]])
# # plt.show()
# plt.savefig('returns_pairplot_rm_outlier.png', dpi=1200)