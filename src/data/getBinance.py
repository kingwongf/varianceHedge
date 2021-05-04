from binance.client import Client
from pathlib import Path
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199


api_key, api_secret = Path('keys/binance').read_text().split(',')





def _get_hist(symb="BTCGBP"):
	client = Client(api_key, api_secret)

	klines = client.get_historical_klines(symb, Client.KLINE_INTERVAL_1MINUTE, "365 days ago UTC", limit=1000)
	df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
	                                   'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
	                                   'Taker buy quote asset volume', 'Ignore'])
	df.set_index(['Open time'], inplace=True)
	df.index = pd.to_datetime(df.index, unit='ms')
	return df

for symb in ["btcgbp", "ethgbp", "xrpgbp", "ltcgbp", "gbpusd", "gbpeur", "linkgbp", "omggbp"]:
	try:
		df = _get_hist(symb=symb.upper())
		df.to_pickle(f'datasets/binance_{symb}.pkl')
	except Exception as e:
		print(e)