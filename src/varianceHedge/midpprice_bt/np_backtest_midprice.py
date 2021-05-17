import pandas as pd
import numpy as np
from src.varianceHedge.tools.np_op_w import compute_op_w
from tqdm import tqdm


pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)


init_port_val = 10000

w_threshold = 0.20
price_sampling_size = 61

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
# st_bt = pd.to_datetime('2020-11-30 06:00:00')


idt_st_bt = mids[mids['Open time']==st_bt].index[0]

## symbs plus date int column
hist_mid = mids[symbs_map.keys()][:idt_st_bt].values


hist_p = np.zeros((mids.shape[0], len(port_map)))
hist_w = {port: np.zeros((hist_mid.shape[0], len(symbs_map))) for port in ports}
hist_n = {port: np.zeros((hist_mid.shape[0], len(symbs_map))) for port in ports}



for idt in tqdm(range(idt_st_bt, mids.shape[0])):
    # print(hist_mid.shape)
    # print(mids[symbs_map.keys()].values[idt, :].reshape(1, -1).shape)
    hist_mid = np.append(hist_mid,
                         mids[symbs_map.keys()].values[idt, :].reshape(1, -1),
                         axis=0)
    # print(hist_mid[-1])
    for port_name, iport in port_map.items():
        # print(port_name)
        # print(f": idt {idt}")
        if idt==idt_st_bt:
            # print('idt==idt_st_bt')
            # print(f"idt: {idt}")

            sample_ret = np.diff(hist_mid[-price_sampling_size:], axis=0) / hist_mid[-price_sampling_size:][:-1, :]
            sample_ret = sample_ret[:, ~np.isnan(sample_ret).all(axis=0)]
            sample_cov = np.cov(sample_ret, rowvar=False)
            exp_ret = sample_ret.mean(axis=0)


            # print('hist_w[port_name]')
            #
            #
            # print(hist_w[port_name].shape)
            # print(getattr(compute_op_w, f"compute_{port_name}_weights")(cov=sample_cov, exp_ret=exp_ret).reshape(1,-1))



            hist_w[port_name] = np.append( hist_w[port_name],
                                           getattr(compute_op_w, f"compute_{port_name}_weights")(cov=sample_cov, exp_ret=exp_ret).reshape(1,-1),
                                           axis=0)


            hist_n[port_name] = np.append(hist_n[port_name],
                                          (init_port_val * hist_w[port_name][idt] / hist_mid[-1]).reshape(1,-1),
                                          axis=0)
            # print(hist_n[port_name][idt])

        curr_port_val = np.sum(hist_n[port_name][-1] * hist_mid[idt])
        hist_p[idt, iport] = curr_port_val
        # print(f"port {port_name} curr_port_val")
        # print(hist_p[idt, iport])


        sample_ret = np.diff(hist_mid[-price_sampling_size:], axis=0) / hist_mid[-price_sampling_size:][:-1, :]
        sample_ret = sample_ret[:, ~np.isnan(sample_ret).all(axis=0)]
        sample_cov = np.cov(sample_ret, rowvar=False)

        exp_ret = sample_ret.mean(axis=0)


        if np.isnan(exp_ret).any():
            print("EXP RET is NAN")
            print('Returns Series')
            print(sample_ret)
            print('Hist Price Series')
            print(hist_mid[-price_sampling_size:])

        if (sample_cov == 0).all():
            opt_w = hist_w[port_name][-1]
        else:
            try:
                opt_w = getattr(compute_op_w, f"compute_{port_name}_weights")(cov=sample_cov, exp_ret=exp_ret,
                                                                       idt=idt, idt_st_bt=idt_st_bt)
            except Exception as e:
                print(e)
                print('cov det')
                print(sample_cov)
                print(np.linalg.det(sample_cov))
                raise e

        ## sort by diff, making always selling first before buying in rebalancing

        # print(opt_w)
        diff_w = np.sort(opt_w - hist_w[port_name][-1])
        symb_sorted = np.argsort(opt_w - hist_w[port_name][-1])

        dw_dict = dict(zip(symb_sorted, diff_w))




        _rebalancing_new_n = {}
        if np.abs(diff_w).max() > w_threshold:
            # print(f"rebalancing for {port_name}")
            for sym_i, diff_w_i in dw_dict.items():
                n_i = diff_w_i * curr_port_val / hist_mid[idt, sym_i]
                _rebalancing_new_n[sym_i] = hist_n[port_name][-1, sym_i] + n_i

        if bool(_rebalancing_new_n):
            hist_n[port_name] = np.append(hist_n[port_name],
                                          np.fromiter(dict(sorted(_rebalancing_new_n.items())).values(), dtype=float).reshape(1,-1),
                                          axis=0)
        # print(f'hist_n {port_name}')
        # print(hist_n[port_name])
        #
        # print(f'hist_w {port_name}')
        # print(hist_w[port_name])


        hist_w[port_name] = np.append(hist_w[port_name],
                                      (hist_n[port_name][-1] * hist_mid[idt] / np.sum(hist_n[port_name][-1] * hist_mid[idt])).reshape(1,-1),
                                      axis=0)




hist_p.to_pickle(f'results/binance/hist_ports_{price_sampling_size}M.pkl')