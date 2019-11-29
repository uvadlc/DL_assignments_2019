
import pandas as pd
import numpy as np
import os
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02
OPTIMIZER_CHOICE='Adam'

data=[['DNN_HIDDEN_UNITS_DEFAULT',  DNN_HIDDEN_UNITS_DEFAULT], ['LEARNING_RATE_DEFAULT', LEARNING_RATE_DEFAULT],['MAX_STEPS_DEFAULT',MAX_STEPS_DEFAULT]]

data=[ [DNN_HIDDEN_UNITS_DEFAULT,  LEARNING_RATE_DEFAULT, NEG_SLOPE_DEFAULT, BATCH_SIZE_DEFAULT, OPTIMIZER_CHOICE]]

df= pd.DataFrame(data, columns=['DNN_HIDDEN_UNITS_DEFAULT','LEARNING_RATE_DEFAULT','NEG_SLOPE_DEFAULT', 'BATCH_SIZE_DEFAULT','OPTIMIZER_CHOICE' ])
print(df)

#df= pd.DataFrame(DNN_HIDDEN_UNITS_DEFAULT, LEARNING_RATE_DEFAULT,MAX_STEPS_DEFAULT)

#df.to_csv(r'c:\pytorch_results\results_pytorch.txt', header=None, index=None, sep=' ', mode='a')

folder = "./pytorch_results/"
f='results_pytorch.txt'
#np.savetxt(folder + f, df, fmt='%s')

df.to_csv(folder + f, header=True, index=None, mode='a', sep=' ')