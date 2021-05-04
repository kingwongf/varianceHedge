import os
import pandas as pd
import numpy as np
from functools import reduce




sym = 'bid' # ask

if sym is 'mid':
    price_cols = [f"{price}[{i}].price" for i in range(0,5) for price in ['bids','asks']]
    vol_cols = [f"{price}[{i}].amount" for i in range(0,5) for price in ['bids','asks']]
else:
    price_cols = [f"{price}[{i}].price" for i in range(0,5) for price in [f'{sym}s']]
    vol_cols = [f"{price}[{i}].amount" for i in range(0,5) for price in [f'{sym}s']]

dts = pd.date_range(start='11/1/2020', end='04/1/2021', freq='MS').strftime("%Y-%m-%d")
files = next(os.walk('datasets'))[2]
done_dates = []


for dt in dts:
    dt_files = [file for file in files if file.endswith('gz') and dt in file]
    dfs = []
    for file in dt_files:
        df = pd.read_csv(f"datasets/{file}", compression='gzip', index_col=['timestamp']).fillna(0)
        df.index = pd.to_datetime(df.index, unit='us')
        if sym == 'mid':
            df[f"best_{df.symbol.iloc[0]}"] = (df["asks[0].price"] + df["bids[0].price"]) / 2
            df[f"weighted_{df.symbol.iloc[0]}"] = np.divide(np.sum(df[price_cols].values * df[vol_cols].values, axis=1),
                                                            (np.sum(df[vol_cols], axis=1)), axis=1)
        else:
            df[f"best_{df.symbol.iloc[0]}"] = df[f"{sym}s[0].price"]
            df[f"weighted_{df.symbol.iloc[0]}"] = np.divide(np.sum(df[price_cols].values * df[vol_cols].values, axis=1),
                                                            (np.sum(df[vol_cols], axis=1)), axis=1)
        dfs.append(df[[f"best_{df.symbol.iloc[0]}", f"weighted_{df.symbol.iloc[0]}"]])
        del df
    df = reduce(lambda X, x: pd.merge(X, x, how='outer', left_index=True, right_index=True), dfs)

    df.to_pickle(f"datasets/consolidated_bitstamp_book_{dt}_{sym}_snapshot_5.pkl")
    # df.to_csv(f"datasets/consolidated_bitstamp_book_{dt}_{sym}_snapshot_5.csv")

#
# for root, _, files in os.walk(".", topdown=False):
#    # print(files)
#     for name in files:
#         dfs = []
#         dt = name.split('_')[-2:-1][0]
#         if name.endswith('gz') and dt not in done_dates:
#             df = pd.read_csv(os.path.join(root, name), compression='gzip', index_col=['timestamp'])
#             df.index = pd.to_datetime(df.index, unit='us')
#             if sym=='mid':
#                 df[f"best_{df.symbol.iloc[0]}"] = (df["asks[0].price"] + df["bids[0].price"])/2
#                 df[f"weighted_{df.symbol.iloc[0]}"] = np.divide(np.sum(df[price_cols].values * df[vol_cols].values, axis=1), (np.sum(df[vol_cols], axis=1)), axis=1)
#             else:
#                 df[f"best_{df.symbol.iloc[0]}"] = df[f"{sym}s[0].price"]
#                 df[f"weighted_{df.symbol.iloc[0]}"] = np.divide(np.sum(df[price_cols].values * df[vol_cols].values, axis=1), (np.sum(df[vol_cols], axis=1)), axis=1)
#             dfs.append(df[[f"best_{df.symbol.iloc[0]}", f"weighted_{df.symbol.iloc[0]}"]])
#             del df
#             done_dates.append(dt)
#
#
#         df = reduce(lambda X,x : pd.merge(X,x, how='outer', left_index=True, right_index=True), dfs)
#
#         # df.to_pickle(f"datasets/consolidated_bitstamp_book_{sym}_snapshot_5.pkl")
#         df.to_csv(f"datasets/consolidated_bitstamp_book_{dt}_{sym}_snapshot_5.csv")