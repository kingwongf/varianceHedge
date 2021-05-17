import pandas as pd
pd.set_option("display.precision", 8)
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from src.varianceHedge.tools.bt_results import stats
from tqdm import tqdm
import ray

pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


mids = pd.read_pickle('datasets/consolidated_binance.pkl').astype('float64').ffill().resample('1min').last().ffill()


port = pd.read_pickle("results/binance/hist_ports_61M.pkl")
ray_port_61M = pd.read_pickle("results/binance/ray_hist_ports_61M.pkl")

ray_port_61M.index = mids.index
st_bt = pd.to_datetime('2020-09-01 00:00:00')
ray_port_61M = ray_port_61M.loc[st_bt:]
print(ray_port_61M)
stats(ray_port_61M)

# fig = ray_port_61M.plot()
# fig.show()

