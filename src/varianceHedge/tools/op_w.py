import numpy as np
import pandas as pd
from pypfopt.cla import CLA
from src.varianceHedge.tools import risk_parity
class compute_op_w:
    def compute_MV_weights(**kwargs):
        try:
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))
        except Exception as e:

            new_vcv, dropped_index,_ = compute_op_w._correct_vcv(kwargs['cov'])
            inv_covar = np.linalg.inv(new_vcv)
            u = np.ones(len(new_vcv))

            w = np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))

            w = pd.Series(w)
            for i in dropped_index:
                w[i] = 0
            return w

    # def compute_MS_weights(**kwargs):
    #     if not compute_op_w._is_pos_def(kwargs['cov']):
    #         kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
    #         kwargs['exp_ret'] = kwargs['exp_ret'].drop('weighted_OMGGBP')
    #
    #     inv_covar = np.linalg.inv(kwargs['cov'])
    #     u = np.ones(len(kwargs['cov']))
    #     w = np.dot(inv_covar, kwargs['exp_ret']) / np.dot(u.T, np.dot(inv_covar, kwargs['exp_ret']))
    #
    #     w = w if w.shape[0]==6 else np.append(w, 0)
    #    return w
    @staticmethod
    def _is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    @staticmethod
    def _correct_vcv(vcv):
        to_drop = vcv.columns[(vcv == 0).all(axis=1)]
        new_vcv = vcv.drop(to_drop, axis=1).drop(to_drop, axis=0)

        return new_vcv, [vcv.index.get_loc(i) for i in to_drop.values], to_drop


    def compute_NRP_weights(**kwargs):
        vcv = kwargs['cov']

        # print((vcv==0).all(axis=1).any())
        new_vcv, dropped_index, _ = compute_op_w._correct_vcv(vcv) if (vcv==0).all(axis=1).any() else (vcv, None, None)
        # print(new_vcv)
        # print(dropped_index)

        weights = 1 / np.diag(new_vcv)
        weights /= sum(weights)

        if dropped_index is None:
            return weights
        else:
            weights = pd.Series(weights)
            for i in dropped_index:
                weights[i] = 0
            return weights

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]

    def compute_MS_weights(**kwargs):
        try:
            w = pd.Series(CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()).loc[kwargs['exp_ret'].index].values
            return w
        except Exception as e:
            new_vcv, dropped_index, to_drop = compute_op_w._correct_vcv(kwargs['cov'])

            new_exp_ret = kwargs['exp_ret'].drop(index =to_drop)

            w = CLA(new_exp_ret, new_vcv, weight_bounds=(-1, 1)).max_sharpe()
            w.update({drop: 0 for drop in to_drop})

            return pd.Series(w)

    def compute_RP_weights(**kwargs):
        return risk_parity.risk_parity_weighting(vcv_matrix=kwargs['cov'], risk_budget=kwargs.get('risk_budget','equal'))