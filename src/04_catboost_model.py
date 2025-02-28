"""This script fits a model using CatBoost, saves the fitted model, and outputs the MAE values as a CSV file.

Usage: src/04_catboost_model.py --train=<train> --test=<test> [--model_path=<model_path>] [--out_path=<out_path>]

Options:
--train=<train>             The path (including filename) of the train dataset file in zip/csv format relative to the root 
--test=<test>               The path (including filename) of the test dataset file in zip/csv format relative to the root
[--model_path=<model_path>] The directory name ending with "/" relative to the root to write the fitted model
[--out_path=<out_path>]     The directory name ending with "/" relative to the root to write the results as a CSV file 
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

def main(train, test, model_path="src/joblib/", out_path="results/"):
    
 
    #create datasets
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    X_train = train_data.drop(columns = ['unacast_session_count'])
    X_test = test_data.drop(columns = ['unacast_session_count'])
    y_train = train_data['unacast_session_count']
    y_test = test_data['unacast_session_count']
    
    #fitting with best params from hyperparameter optimization

    catboost = CatBoostRegressor(verbose=False, loss_function='MAE', iterations=201, depth=13, bagging_temperature=6) 

    catboost.fit(X_train, y_train)
    
    y_preds_train = catboost.predict(X_train)
    y_preds_test = catboost.predict(X_test)

    #capping the negative predicted values with 0
    y_preds_train = list(map(lambda x: 0 if x<0 else x, y_preds_train))
    y_preds_test = list(map(lambda x: 0 if x<0 else x, y_preds_test))
    
    train_mae = mean_absolute_error(y_train, y_preds_train)
    
    test_mae = mean_absolute_error(y_test, y_preds_test)
   
    if model_path is None:
        model_path = "src/joblib/catboost_model.joblib"
    else:
        model_path = model_path+"catboost_model.joblib"
    
    # save the trained model
    dump(catboost, model_path)

    # File to save results
    if out_path is None:
        out_file = "results/catboost_train_result.csv"
    else:
        out_file = out_path+"catboost_train_result.csv"

    #removing existing file
    if os.path.exists(out_file):
        os.remove(out_file)

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
    train_data_loc = "data/dummy/dummy_train_data.zip"
    test_data_loc = "data/dummy/dummy_test_data.zip"
    main(train_data_loc, test_data_loc)
    assert os.path.exists("src/joblib/catboost_model.joblib"), "Model dump not found in location"
    assert os.path.exists("results/catboost_train_result.csv"), "Results file not found in location"
    os.remove("src/joblib/catboost_model.joblib")
    os.remove("results/catboost_train_result.csv")


if __name__ == "__main__":
    test_fun()
    main(opt["--train"], opt["--test"], opt["--model_path"], opt["--out_path"])

