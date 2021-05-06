import pandas as pd
import matplotlib.pyplot as plt
pd.options.plotting.backend = "plotly"
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)



func_w = pd.read_pickle(f'investigate/func_MV.pkl')
func_opt_w = pd.read_pickle(f'investigate/func_opt_MV.pkl')



nonfunc_w = pd.read_pickle(f"investigate/nonfunc_opt_MV.pkl")
nonfunc_opt_w = pd.read_pickle(f"investigate/nonfunc_opt_MV.pkl")

print(func_opt_w)
print(nonfunc_opt_w)

print(func_w)
print(nonfunc_w)