import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from src.varianceHedge.tools.np_op_w import compute_op_w
from tqdm import tqdm
import ray

# pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)

ray.init()

init_port_val = 10000

w_threshold = 0.20
price_sampling_size = 121

ports = ['MV','NRP', 'RP', 'MS', 'UW']



mids = pd.read_pickle('datasets/consolidated_binance.pkl').astype('float64').ffill().resample('1min').last().ffill().reset_index().ffill()

## Transform everything to numpy

# Mapping int to symbs and port
symbs_map = {'btcgbp':0, 'ethgbp':1, 'xrpgbp':2}
# symbs_map = dict(zip(range(mids.columns.shape[0]), mids.columns))
port_map = dict(zip(ports, range(len(ports))))

## portfolio value is 10,000 on st_bt

## idt_st_bt = 129960
st_bt = pd.to_datetime('2020-09-01 00:00:00')



idt_st_bt = mids[mids['Open time']==st_bt].index[0]

## symbs plus date int column
hist_mid = mids[symbs_map.keys()][:idt_st_bt].values




@ray.remote
def bt(port):
    hist_p = np.zeros((mids.shape[0], 1))
    hist_mid = mids[symbs_map.keys()][:idt_st_bt].values
    hist_w = np.zeros((hist_mid.shape[0], len(symbs_map)))
    hist_n = np.zeros((hist_mid.shape[0], len(symbs_map)))

    for idt in range(idt_st_bt, mids.shape[0]):
        hist_mid = np.append(hist_mid,
                             mids[symbs_map.keys()].values[idt, :].reshape(1, -1),
                             axis=0)
        if idt == idt_st_bt:
            sample_ret = np.diff(hist_mid[-price_sampling_size:], axis=0) / hist_mid[-price_sampling_size:][1:, :]
            sample_ret = sample_ret[:, ~np.isnan(sample_ret).all(axis=0)]
            sample_cov = np.cov(sample_ret, rowvar=False)
            exp_ret = sample_ret.mean(axis=0)

            hist_w = np.append(hist_w,
                               getattr(compute_op_w, f"compute_{port}_weights")(cov=sample_cov, exp_ret=exp_ret).reshape(1, -1),
                               axis=0)

            hist_n = np.append(hist_n, (init_port_val * hist_w[idt] / hist_mid[-1]).reshape(1, -1), axis=0)

        curr_port_val = np.sum(hist_n[-1] * hist_mid[idt])
        hist_p[idt] = curr_port_val


        sample_ret = np.diff(hist_mid[-price_sampling_size:], axis=0) / hist_mid[-price_sampling_size:][1:, :]
        sample_ret = sample_ret[:, ~np.isnan(sample_ret).all(axis=0)]
        sample_cov = np.cov(sample_ret, rowvar=False)
        exp_ret = sample_ret.mean(axis=0)

        if (sample_cov == 0).all():
            opt_w = hist_w[-1]
        else:
            opt_w = getattr(compute_op_w, f"compute_{port}_weights")(cov=sample_cov, exp_ret=exp_ret,
                                                                      idt=idt, idt_st_bt=idt_st_bt)

        ## sort by diff, making always selling first before buying in rebalancing

        # print(opt_w)
        diff_w = np.sort(opt_w - hist_w[-1])
        symb_sorted = np.argsort(opt_w - hist_w[-1])

        dw_dict = dict(zip(symb_sorted, diff_w))

        _rebalancing_new_n = {}
        if np.abs(diff_w).max() > w_threshold:
            # print(f"rebalancing for {port_name}")
            for sym_i, diff_w_i in dw_dict.items():
                n_i = diff_w_i * curr_port_val / hist_mid[idt, sym_i]
                _rebalancing_new_n[sym_i] = hist_n[-1, sym_i] + n_i

        if bool(_rebalancing_new_n):
            hist_n = np.append(hist_n,
                               np.fromiter(dict(sorted(_rebalancing_new_n.items())).values(),dtype=float).reshape(1, -1),
                               axis=0)


        hist_w = np.append(hist_w,
                           (hist_n[-1] * hist_mid[idt] / np.sum(hist_n[-1] * hist_mid[idt])).reshape(1, -1),
                           axis=0)


    return hist_p.flatten()


futures = [bt.remote(p) for p in ports]
ps = ray.get(futures)
ps = np.array(ps).T

hist_ports = pd.DataFrame(ps, columns=ports)
hist_ports.to_pickle(f'results/binance/ray_hist_ports_{price_sampling_size}M.pkl')