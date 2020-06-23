"""This script plots the residual plots of the 3 models we run in our pipeline (LightGBM, CatBoost and GradientBoostingRegressor)

Usage: src/06_error_plots.py --train=<train> --test=<test> --model_path=<model_path> --out_path=<out_path>

Options:
--train=<train>             The path (including filename) of the train dataset file in zip/csv format relative to the root 
--test=<test>               The path (including filename) of the test dataset file in zip/csv format relative to the root
--model_path=<model_path>   The directory name ending with "/" relative to the root to find the fitted model
--out_path=<out_path>       The directory name ending with "/" relative to sve the different plots
"""

from docopt import docopt
import os
import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load
import csv
import altair as alt

opt = docopt(__doc__)

def main(train, test, model_path="src/joblib/", out_path="results/report_figures/"):
    
 
    #create datasets
    train_data = pd.read_csv(train)
    test_data = pd.read_csv(test)

    X_train = train_data.drop(columns = ['unacast_session_count'])
    X_test = test_data.drop(columns = ['unacast_session_count'])
    y_train = train_data['unacast_session_count']
    y_test = test_data['unacast_session_count']
    
    #loading models
    filepath = model_path+"lgbm_model.joblib" 
    lgbm = load(filepath)

    filepath = model_path+"gbr_model.joblib" 
    xgb = load(filepath)

    filepath = model_path+"catboost_model.joblib" 
    catb = load(filepath)
    
    for name, model_try in {'lgbm':lgbm, 'xgb':xgb, 'catb':catb}.items():
        plots = plot_resid(model_try, X_train=X_train, y_train=y_train, X_valid=X_test, y_valid=y_test)
        plots['Train'].save(out_path+ '/' +name+'_train.html')
        plots['Valid'].save(out_path+ '/' +name+'_test.html')


def test_fun():
    """
    This functions checks if the main function is able to generate a plot, and save it

    """
    train_data_loc = "data/dummy/dummy_train_data.zip"
    test_data_loc = "data/dummy/dummy_test_data.zip"
    main(train_data_loc, test_data_loc, 'src/joblib/', 'results/report_figures')
    assert os.path.exists("results/report_figures/lgbm_train.html"), 'LGBM train plot not found in location'
    assert os.path.exists("results/report_figures/catb_test.html"), 'Catboost test plot not found in location'
    os.remove("results/report_figures/lgbm_train.html")
    os.remove("results/report_figures/xgb_train.html")
    os.remove("results/report_figures/catb_train.html")
    os.remove("results/report_figures/lgbm_test.html")
    os.remove("results/report_figures/xgb_test.html")
    os.remove("results/report_figures/catb_test.html")


def plot_resid(model, X_train=None, y_train=None, X_valid=None, y_valid=None, plot = 'both'):
    """
    Creates scatter plot for train and validation data for a given model.
    
    Parameters
    ----------
    model: A trained model type object with a model.predict method
    X_train: pd.DataFrame, `X_train`
    y_train: Array like set of train target variable
    X_valid: pd.DataFrame, `X_valid`
    y_valid: Array like set of validation target variable
    plot: String, if 'valid' won't plot train, if 'train' won't plot valid, else will plot both
    
    Returns
    -------
    dictionary with the two plots, with the keys 'Train' and 'Valid' respectively
    """
    d = dict()
    
    if plot != 'valid':
        train_df = pd.DataFrame({'Predicted Train':list(map(lambda x: 0 if x<0 else x, model.predict(X_train))), 'True Train':y_train})
        train_df['Train Error'] =  train_df['Predicted Train'] - train_df['True Train']
        train_dist = alt.Chart(train_df).mark_circle().encode(alt.X("True Train:Q"), y=alt.Y('Train Error:Q'))
        d["Train"] = train_dist
    else:
        d["Train"] = "No Validation set given"
        
        
    if plot != 'train':
        valid_df = pd.DataFrame({'Predicted Valid':list(map(lambda x: 0 if x<0 else x, model.predict(X_valid))), 'True Valid':y_valid})
        valid_df['Valid Error'] =  valid_df['Predicted Valid'] - valid_df['True Valid']
        
        valid_dist = alt.Chart(valid_df).mark_circle().encode(alt.X("True Valid:Q"), y=alt.Y('Valid Error Distance:Q'))
        d["Valid"] = valid_dist
    else:
        d["Valid"] = "No training set given"
        
    return d

if __name__ == "__main__":
    test_fun()
    main(opt["--train"], opt["--test"], opt["--model_path"], opt["--out_path"])
