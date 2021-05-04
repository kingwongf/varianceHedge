import plotly.express as px
import pandas as pd

df = pd.read_pickle('main_hist_p.pkl')
melted_df = df.melt(id_vars='w_threshold', value_vars=['MV','RP','MS','UW'], var_name='portfolio', value_name='value', ignore_index=False)
fig = px.line(melted_df, x=melted_df.index, y="value", animation_frame="w_threshold", color="portfolio")
fig["layout"].pop("updatemenus") # optional, drop animation buttons

fig.write_html("opt_w_backtest.html")
fig.show()
