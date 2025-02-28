
"""This script generates predictions given new, preprocessed data and pre-fitted model.

Usage: src/07_prediction.py --new_data=<new_data> [--model_path=<model_path>] [--out_path=<out_path>]

Options:
--new_data=<new_data>       The path (including filename) of the new data file in zip/csv format relative to the root 
[--model_path=<model_path>] The directory name ending with "/" of the folder where fitted models are located
[--out_path=<out_path>]     The directory name ending with "/" relative to root where to write the zip/csv with new_data and predicted target values from all 3 models named predicted_data.zip
"""

from docopt import docopt
import os
import pandas as pd
import numpy as np
from joblib import dump, load

opt = docopt(__doc__)

def main(new_data, model_path="src/joblib/", out_path="results/"):
    
 
    #create datasets
    data = pd.read_csv(new_data)
    #checking if prediction data already contains target
    if "unacast_session_count" in set(data.columns):
        y = data.loc[:, 'unacast_session_count']
        predict_data = data.drop(columns=['unacast_session_count'])
    else:
        predict_data = data
    
    if model_path is None:
        model_path = "src/joblib/"

    #loading models
    filepath = model_path+"lgbm_model.joblib" 
    lgbm = load(filepath)

    filepath = model_path+"gbr_model.joblib" 
    xgb = load(filepath)

    filepath = model_path+"catboost_model.joblib" 
    catb = load(filepath)

    # getting predictions from saved models

    y_preds_lgbm = lgbm.predict(predict_data)
    y_preds_xgb = xgb.predict(predict_data)
    y_preds_catb = catb.predict(predict_data)

    # capping the negative predicted values with 0
    y_preds_lgbm = list(map(lambda x: 0 if x<0 else x, y_preds_lgbm))
    y_preds_xgb = list(map(lambda x: 0 if x<0 else x, y_preds_xgb))
    y_preds_catb = list(map(lambda x: 0 if x<0 else x, y_preds_catb))
    
    # making session counts as integers
    y_preds_lgbm = list(map(lambda x: int(x), y_preds_lgbm))
    y_preds_xgb = list(map(lambda x: int(x), y_preds_xgb))
    y_preds_catb = list(map(lambda x: int(x), y_preds_catb))
    
    #adding the predicted values to dataframe

    predict_data['session_count_lightgbm'] = y_preds_lgbm
    predict_data['session_count_xgboost'] = y_preds_xgb
    predict_data['session_count_catboost'] = y_preds_catb
    if "unacast_session_count" in set(data.columns):
        predict_data['unacast_session_count'] = y
  
    if out_path is None:
        out_path = "results/predicted_data.zip"
    else:
        out_path = out_path+"predicted_data.zip"
    
    #removing existing file
    if os.path.exists(out_path):
        os.remove(out_path)
    
    compression_opts = dict(method='zip',archive_name='predicted_data.csv')  
    predict_data.to_csv(out_path, index = False, compression=compression_opts)


def test_fun():
    """
    This functions checks if the main function is able to predict from a stored model and put the results in zip/csv file.

    """
    pred_data_loc = "data/dummy/dummy_pred_data.zip"
    main(pred_data_loc)
    assert os.path.exists("results/predicted_data.zip"), "Predicted file not found in location"
    os.remove("results/predicted_data.zip")


if __name__ == "__main__":
    test_fun()
    main(opt["--new_data"], opt["--model_path"], opt["--out_path"])
