
"""This script runs lgbm and stores the fitted model and output the MAE scores into a csv. 
Usage: scripts/lgbm_model.py --train=<train> --test=<test> [--out_path=<out_path>]
Options:
--train=<train>            The location of the train dataset file in zip/csv format relative to the root 
--test=<test>              The location of the test dataset file in zip/csv format relative to the root
[--out_path=<out_path>]    Path (folder_name) relative to root where to write the fitted model
"""

from docopt import docopt
import requests
import os
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

opt = docopt(__doc__)

def main(train, test, out_path = ''):
    
    #create datasets
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)
    
    X_train = train_data.drop(columns = ['unacast_session_count'])
    X_test = test_data.drop(columns = ['unacast_session_count'])
    y_train = train_data['unacast_session_count']
    y_test = test_data['unacast_session_count']
    
    #fitting with best params from hyperparameter optimization
    
    lgbm = LGBMRegressor(learning_rate = 0.19981387712135354 , max_dept = 105.0, num_leaves = 300, random_state = 2020) 

    lgbm.fit(X_train, y_train)
    
    y_preds_train = lgbm.predict(X_train)
    y_preds_test = lgbm.predict(X_test)

    #capping the negative predicted values with 0
    y_preds_train = list(map(lambda x: 0 if x<0 else x, y_preds_train))
    y_preds_test = list(map(lambda x: 0 if x<0 else x, y_preds_test))
    
    print('\nLGBM scores train: ')
    train_mae = mean_absolute_error(y_train, y_preds_train)
    print("Mean absolute error: %0.3f" % mae)
    
    print('\nLGBM scores test: ')
    test_mae = mean_absolute_error(y_valid, y_preds_test)
    print("Mean absolute error: %0.3f" % mae)
    
        
    # save the model to disk
    model_path = out_path+'lgbm_model.joblib'
    dump(lgbm, model_path)

    # File to save results
    out_file = 'results/lgbm_train_result.csv'
    connection = open(out_file, 'w') #creating file by truncating previous one
    writer = csv.writer(connection)

    # Write the headers to the file
    writer.writerow(['model', 'train mae', 'test mae'])
    writer.writerow(["LightGBM", train_mae, test_mae])

    of_connection.close()

def test_fun():
    """
    This functions checks if the main function is able to fit a model and store the results in csv file.

    """
    train_data = "train_data.zip"
    test_data = "valid_data.zip"
    main(train_data, test_data)
    assert os.path.exists("lgbm_model.joblib"), "File not found in location"


if __name__ == "__main__":
    test_fun()
    main(opt["--train"], opt["--test"], opt["--out_path"])
