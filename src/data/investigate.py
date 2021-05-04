import pandas as pd
import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

month_dt = '2021-01-01'
mids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_mid_snapshot_5.pkl').ffill().resample('1min').last()
# bids = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_bid_snapshot_5.pkl').ffill().resample(
# 	'1min').last()
# asks = pd.read_pickle(f'datasets/consolidated_bitstamp_book_{month_dt}_ask_snapshot_5.pkl').ffill().resample(
# 	'1min').last()


omggbp_df = pd.read_csv('datasets/bitstamp_book_snapshot_25_2021-01-01_OMGGBP.csv.gz', compression='gzip', index_col=['timestamp'])

omggbp_df.index = pd.to_datetime(omggbp_df.index, unit='us')


omggbp_df.sort_index(inplace=True)
# rint(omggbp_df.loc['2021-01-01 0055:00':][['exchange', 'symbol', 'local_timestamp', 'asks[0].price', 'asks[0].amount', 'bids[0].price', 'bids[0].amount']].head(20))
print(mids.loc['2021-01-01 06:05:00 ':].head(20).pct_change())

# fig = (mids.pct_change().fillna(0)+1).cumprod().plot()
# fig.show()



# fig = mids[mids.columns[mids.columns.str.contains('OMG')]].plot()
# fig.show()