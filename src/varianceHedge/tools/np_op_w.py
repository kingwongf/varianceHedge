import numpy as np
from pypfopt.cla import CLA
from src.varianceHedge.tools import np_risk_parity
class compute_op_w:
    def compute_MV_weights(**kwargs):
        try:
            inv_covar = np.linalg.inv(kwargs['cov'])
            u = np.ones(len(kwargs['cov']))
            return np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))
        except Exception as e:

            new_vcv, dropped_index,new_symbs_li = compute_op_w._correct_vcv(kwargs['cov'])


            inv_covar = np.linalg.inv(new_vcv)
            u = np.ones(len(new_vcv))

            w = np.dot(inv_covar, u) / np.dot(u.T, np.dot(inv_covar, u))



            w = compute_op_w._adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index)
            return np.fromiter(w.values(), dtype=float)

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
        index_to_drop = np.flatnonzero((vcv == 0).all(axis=1))

        new_vcv = np.delete(vcv, index_to_drop, axis=1)
        new_vcv = np.delete(new_vcv, index_to_drop, axis=0)
        # print(new_vcv)
        # print('index_to_drop')
        # print(index_to_drop)

        new_symbs_li = list(range(len(vcv)))
        for d in index_to_drop:
            new_symbs_li.remove(d)

        return new_vcv, index_to_drop, new_symbs_li

    @staticmethod
    def _adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index):
        w = dict(zip(new_symbs_li, w))
        w = dict(sorted(w.items()))
        for i in dropped_index:
            w[i] = 0
        w = dict(sorted(w.items()))
        return w


    def compute_NRP_weights(**kwargs):
        vcv = kwargs['cov']

        new_vcv, dropped_index, new_symbs_li = compute_op_w._correct_vcv(vcv) if (vcv==0).all(axis=1).any() else (vcv, None, None)



        w = 1 / np.diag(new_vcv)
        w /= sum(w)

        if dropped_index is None:
            return w
        else:



            # w = dict(zip(new_symbs_li, w))
            # w = dict(sorted(w.items()))
            # for i in dropped_index:
            #     w[i] = 0
            #
            # w = dict(sorted(w.items()))

            w = compute_op_w._adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index)

            return np.fromiter(w.values(), dtype=float)

    def compute_UW_weights(**kwargs):
        return np.array([1/kwargs['cov'].shape[0]]*kwargs['cov'].shape[0])

    def compute_MS_weights(**kwargs):

        if (kwargs['cov'] == 0).all(axis=1).any():
            new_vcv, dropped_index, new_symbs_li = compute_op_w._correct_vcv(kwargs['cov'])

            new_vcv = new_vcv + np.eye(new_vcv.shape[0]) * 1e-9 if np.linalg.det(new_vcv) <= 1e-23 else new_vcv

            exp_ret = kwargs['exp_ret']


            new_exp_ret = np.delete(exp_ret, dropped_index)
            w = CLA(new_exp_ret, new_vcv, weight_bounds=(-5, 5)).max_sharpe().values()
            # try:
            #     w = CLA(new_exp_ret, new_vcv, weight_bounds=(-1, 1)).max_sharpe().values()
            # except Exception as e:
            #     try:
            #         w = CLA(new_exp_ret, new_vcv, weight_bounds=(-5, 5)).max_sharpe().values()
            #     except Exception as e:
            #         inv_covar = np.linalg.inv(new_vcv)
            #         u = np.ones(len(new_vcv))
            #         w = np.dot(inv_covar, exp_ret) / np.dot(u.T, np.dot(inv_covar, exp_ret))

            w = compute_op_w._adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index)
            return np.fromiter(w.values(), dtype=float)

        else:

            kwargs['cov'] = kwargs['cov'] + np.eye(kwargs['cov'].shape[0]) * 1e-9 if np.linalg.det(kwargs['cov']) <= 1e-23 else kwargs['cov']

            w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-5, 5)).max_sharpe()
            # try:
            #     w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-5, 5)).max_sharpe()
            # except Exception as e:
            #     w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-1, 1)).max_sharpe()
            return np.fromiter(w.values(), dtype=float)




        # try:
        #     w = CLA(kwargs['exp_ret'], kwargs['cov'], weight_bounds=(-5, 5)).max_sharpe()
        #     # print(f'w MS: {np.fromiter(w.values(), dtype=float)}')
        #     return np.fromiter(w.values(), dtype=float)
        # except Exception as e:
        #     try:
        #         new_vcv, dropped_index, new_symbs_li = compute_op_w._correct_vcv(kwargs['cov'])
        #         exp_ret= kwargs['exp_ret']
        #         new_exp_ret = np.delete(exp_ret, dropped_index)
        #         w = CLA(new_exp_ret, new_vcv, weight_bounds=(-5, 5)).max_sharpe().values()
        #         w = compute_op_w._adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index)
        #
        #         # w = dict(zip(new_symbs_li, CLA(new_exp_ret, new_vcv, weight_bounds=(-1, 1)).max_sharpe().values()))
        #         # w = dict(sorted(w.items()))
        #         # for i in dropped_index:
        #         #     w[i] = 0
        #         # w = dict(sorted(w.items()))
        #         #
        #         # print(f'MS w: {w}')
        #         return np.fromiter(w.values(), dtype=float)
        #     except Exception as e:
        #         try:
        #             new_vcv, dropped_index, new_symbs_li = compute_op_w._correct_vcv(kwargs['cov'])
        #             exp_ret = kwargs['exp_ret']
        #             new_exp_ret = np.delete(exp_ret, dropped_index)
        #             w = CLA(new_exp_ret, new_vcv, weight_bounds=(-1, 1)).max_sharpe().values()
        #             w = compute_op_w._adding_zero_vcv_zero_weights(w, new_symbs_li, dropped_index)
        #             return np.fromiter(w.values(), dtype=float)
        #         except Exception as e:
        #             print('new_exp_ret')
        #             print(new_exp_ret)
        #             print('new_vcv')
        #             print(new_vcv)
        #             print("ERRUR")
        #             print(e)
        #             raise e

    def compute_RP_weights(**kwargs):
        return np_risk_parity.risk_parity_weighting(vcv_matrix=kwargs['cov'], risk_budget=kwargs.get('risk_budget','equal'))