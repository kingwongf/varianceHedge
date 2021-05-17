import pandas as pd
import plotly.express as px

mids = pd.read_pickle('datasets/consolidated_binance.pkl').astype('float64').ffill().resample('1min').last()
print(mids.notna())

print(mids.notna().loc['2020-11-30 05:59:00':])
fig = px.imshow(mids.notna())
fig.show()