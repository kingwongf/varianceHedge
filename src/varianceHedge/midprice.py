import pandas as pd
import numpy as np
import seaborn as sns
pd.options.plotting.backend = "plotly"
from scipy import stats
import matplotlib.pyplot as plt


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


df = pd.read_pickle('datasets/consolidated_bitstamp_book_mid_snapshot_5.pkl').ffill().resample('1min').last()
df = df[df.columns[df.columns.str.startswith('weighted_')]]
rets = df.pct_change().dropna()

sym = rets.columns

## min. variance hedge
## variances-covariances sampling period 1hr

rolling_60_mins_cov_df = rets.rolling('60min', min_periods=60).cov().dropna(axis=0)
rolling_120_mins_cov_df = rets.rolling('120min', min_periods=60).cov().dropna(axis=0)

rolling_resample_5_mins_cov_df = df.pct_change(5).rolling('60min', min_periods=60).cov().dropna(axis=0)
rolling_resample_10_mins_cov_df = df.pct_change(10).rolling('60min', min_periods=60).cov().dropna(axis=0)

rolling_strict_resample_5_mins_cov_df = df.pct_change(5).rolling('60min', min_periods=60).cov().dropna(axis=0)
rolling_strict_resample_10_mins_cov_df = df.pct_change(10).rolling('60min', min_periods=60).cov().dropna(axis=0)


rolling_cov_dfs = [rolling_60_mins_cov_df, rolling_120_mins_cov_df,
                   rolling_resample_5_mins_cov_df, rolling_resample_10_mins_cov_df,
                   rolling_strict_resample_5_mins_cov_df, rolling_strict_resample_10_mins_cov_df]
cov_names = ['rolling 60mins', 'rolling_120mins', 'resample_5mins', 'resample_10mins', 'strict_5mins,', 'strict_10mins']

names_w = ['w_MV', 'w_RP', 'w_UW', 'w_MS']
## dynamic daily minimum weights

vol_df = pd.DataFrame()

# print(rets.iloc[:60])
# print(rets.iloc[:60].cov())

port = pd.DataFrame(index=df.index, columns=pd.MultiIndex.from_product([names_w, cov_names]))
for cov, cov_name in zip(rolling_cov_dfs, cov_names):

    # print(cov_name)
    # print(cov)

    w_MV = cov.groupby(level=0, axis=0).apply(compute_MV_weights).apply(pd.Series)
    w_RP = cov.groupby(level=0, axis=0).apply(compute_RP_weights).apply(pd.Series)
    w_UW = pd.DataFrame((1/rets.shape[1])*np.ones(w_MV.shape), index=w_MV.index)
    w_MV.columns = w_RP.columns = w_UW.columns = df.columns



    exp_rets = rets.rolling('60min').mean()

    w_MS = pd.DataFrame(index=w_MV.index, columns= df.columns)
    for dt, cov in cov.dropna(how='any').groupby(level=0):
        w_MS.loc[dt] = compute_MS_weights(exp_rets.loc[dt], cov)



    for w, name_w in zip([w_MV, w_RP, w_UW, w_MS], names_w):
        # print(name_w)
        # print(w.shift(1))
        # print(rets)
        w = w.shift(1)

        # print(f"{cov_name}: {name_w}")
        # print(w)
        ret_p = (w*rets[sym].loc[w.index]).sum(axis=1)

        # print(ret_p)
        ret_p.iloc[0] = 0

        port[(name_w, cov_name)] = (ret_p + 1).cumprod()

        vol = '{:f}'.format(ret_p.std())
        vol_df.loc[name_w, cov_name] = ret_p.std()
        # print(f"{name_w} volatility: {vol}")

    # break

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
    # print(vol_df)
    # port[['w_MV', 'w_RP', 'w_UW']].plot()
    # plt.show()

# print(port)
idx = pd.IndexSlice
# port.columns.get_level_values(0)
# fig = port.loc[:, (port.columns.get_level_values(0)=='w_MV' | port.columns.get_level_values(0)=='w_RP') ].plot()
plot_port = port[['w_MV' ,'w_RP']]
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