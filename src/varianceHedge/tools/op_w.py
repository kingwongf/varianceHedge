import numpy as np
import pandas as pd
from pypfopt.cla import CLA

class compute_op_w:
    def compute_MV_weights(**kwargs):
        try:
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))
        except Exception as e:
            kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            w = np.append(np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u)), 0)
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

    def compute_RP_weights(**kwargs):
        weights = (1 / np.diag(kwargs['cov']))
        return weights / sum(weights)

    def compute_UW_weights(**kwargs):
        return [1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0]

    def compute_MS_weights(**kwargs):
        try:
            w = pd.Series(CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()).loc[kwargs['exp_ret'].index].values
            return w
        except Exception as e:
            kwargs['cov'] = kwargs['cov'].drop('weighted_OMGGBP', axis=1).drop('weighted_OMGGBP', axis=0)
            kwargs['exp_ret'] = kwargs['exp_ret'].drop('weighted_OMGGBP')

            w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()
            w.update({'weighted_OMGGBP':0})

            return pd.Series(w).loc[kwargs['exp_ret'].index.append(pd.Index(['weighted_OMGGBP']))].values