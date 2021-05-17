import pandas as pd
pd.options.plotting.backend = "plotly"
mids = pd.read_pickle('datasets/consolidated_binance.pkl').astype('float64').ffill().resample('1min').last()
fig = mids['xrpgbp'].plot()
fig.show()