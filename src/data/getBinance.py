from binance.client import Client
from pathlib import Path
import pandas as pd
from functools import reduce
import seaborn
import plotly.express as px
pd.options.plotting.backend = "plotly"
import numpy as np
pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


api_key, api_secret = Path('keys/binance').read_text().split(',')





def _get_kline_hist(symb="BTCGBP"):
	client = Client(api_key, api_secret)

	klines = client.get_historical_klines(symb, Client.KLINE_INTERVAL_1MINUTE, "365 days ago UTC", limit=1000)
	df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
	                                   'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
	                                   'Taker buy quote asset volume', 'Ignore'])
	df.set_index(['Open time'], inplace=True)
	df.index = pd.to_datetime(df.index, unit='ms')
	return df

def kline():
	# for symb in ["btcgbp", "ethgbp", "xrpgbp", "ltcgbp", "gbpusd", "gbpeur", "linkgbp", "omggbp"]:
	for symb in ["btcgbp", "ethgbp", "xrpgbp",]:
		try:
			df = _get_kline_hist(symb=symb.upper())
			df.to_pickle(f'datasets/binance_ohlcv_{symb}.pkl')
		except Exception as e:
			print(symb)
			print(e)

def _consolidate():
	symbs = ["btcgbp", "ethgbp", "xrpgbp", "ltcgbp", "linkgbp"]

	df = reduce(lambda X, x: pd.merge(X, x, how='outer', left_index=True, right_index=True),
	            [pd.read_pickle(f'datasets/binance_ohlcv_{symb}.pkl')['Open'].rename(symb) for symb in symbs])

	# fig = df.astype('float64').ffill().plot()
	# fig.show()
	# print(df.isna().sum())
	# rint(df)
	fig = px.imshow(df)
	fig.show()

	print(df)
	df.to_pickle('datasets/consolidated_binance.pkl')
_consolidate()

# kline()