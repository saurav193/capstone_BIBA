
"""This script runs lgbm and stores the fitted model. 
Usage: scripts/lgbm_model.py --X_train=<X_train> --X_valid=<X_valid> --y_train=<y_train> --y_valid=<y_valid> [--out_path=<out_path>]
Options:
--X_train=<X_train>
--X_valid=<X_valid>
--y_train=<y_train>
--y_valid=<y_valid>
[--out_path=<out_path>]    Path (folder_name) relative to root where to write the fitted model in .pkl format
"""

from docopt import docopt
import requests
import os
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
import pickle

from sklearn.metrics import mean_squared_error, mean_absolute_error

opt = docopt(__doc__)

def main(X_train, X_valid, y_train, y_valid, out_path = ''):
    
    #fitting with best params from hyperparameter optimization
    lgbm = LGBMRegressor(boosting_type = 'goss', learning_rate = 0.199813, max_dept = 105, num_leaves = 300, random_state = 2020) 

    lgbm.fit(X_train, y_train)
    
    y_preds_train = lgbm.predict(X_train)
    y_preds_valid = lgbm.predict(X_valid)
    
    print('LGBM scores training: ')
    rmse = mean_squared_error(y_train, y_preds_train, squared = False)
    mae = mean_absolute_error(y_train, y_preds_train)
    print("Root mean squared error: %0.3f  and  Mean absolute error: %0.3f" % rmse, mae)
    
    print('\nLGBM scores validation: ')
    rmse = mean_squared_error(y_valid, y_preds_valid, squared = False)
    mae = mean_absolute_error(y_valid, y_preds_valid)
    print("Root mean squared error: %0.3f  and  Mean absolute error: %0.3f" % rmse, mae)
    
        
    # save the model to disk
    filename = out_path+'lgbm_model.pkl'
    pickle.dump(lgbm, open(filename, 'wb'))

if __name__ == "__main__":
    
    main(opt["--X_train"], opt["--X_valid"], opt["--y_train"], opt["--y_valid"], opt["--out_path"])
