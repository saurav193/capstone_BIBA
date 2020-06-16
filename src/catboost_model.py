"""This script fits a model using CatBoost, saves the model, and outputs the MAE values as a CSV file.
Usage: src/lgbm_model.py --train=<train> --test=<test> [--out_path=<out_path>]
Options:
--train=<train>            The location(including filename) of the train dataset file in zip/csv format relative to the root 
--test=<test>              The location(including filename) of the test dataset file in zip/csv format relative to the root
[--out_path=<out_path>]    Path(folder_name) ending with "/" relative to root where to write the fitted model
"""

from docopt import docopt
import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from joblib import dump, load
from sklearn.metrics import mean_squared_error, mean_absolute_error
import csv

opt = docopt(__doc__)

def main(train, test, out_path=""):
    
 
    #create datasets
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    X_train = train_data.drop(columns = ['unacast_session_count'])
    X_test = test_data.drop(columns = ['unacast_session_count'])
    y_train = train_data['unacast_session_count']
    y_test = test_data['unacast_session_count']
    
    #fitting with best params from hyperparameter optimization

    catboost = CatBoostRegressor(verbose=False, loss_function='RMSE', iterations=201, depth=12) 

    catboost.fit(X_train, y_train)
    
    y_preds_train = catboost.predict(X_train)
    y_preds_test = catboost.predict(X_test)

    #capping the negative predicted values with 0
    y_preds_train = list(map(lambda x: 0 if x<0 else x, y_preds_train))
    y_preds_test = list(map(lambda x: 0 if x<0 else x, y_preds_test))
    
    train_mae = mean_absolute_error(y_train, y_preds_train)
    
    test_mae = mean_absolute_error(y_test, y_preds_test)
   
    if out_path is None:
        model_path = "src/catboost_model.joblib"
    else:
        model_path = out_path+"catboost_model.joblib"
    
    # save the trained model
    dump(catboost, model_path)

    # File to save results
    out_file = 'results/catboost_train_result.csv'

    #removing existing file
    if os.path.exists("results/catboost_train_result.csv"):
        os.remove('results/catboost_train_result.csv')

    with open(out_file, 'w', newline='') as f: #creating file by truncating previous one
        writer = csv.writer(f)
        # Write the headers to the file
        writer.writerow(['model', 'train mae', 'test mae'])
        writer.writerow(["Catboost", train_mae, test_mae])
    f.close()

def test_fun():
    """
    This functions checks if the main function is able to fit a model and store the results in csv file.
    """
    train_data_loc = "src/catboost_train_data.zip"
    test_data_loc = "src/catboost_test_data.zip"
    main(train_data_loc, test_data_loc)
    assert os.path.exists("src/catboost_model.joblib"), "Model dump not found in location"
    assert os.path.exists("results/catboost_train_result.csv"), "Results file not found in location"


if __name__ == "__main__":
    test_fun()
    main(opt["--train"], opt["--test"], opt["--out_path"])
