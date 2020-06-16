
"""This script predicts the model output from the new data given and fitted model in training pipeline.

Usage: src/07_prediction.py --pred=<pred> [--model_path=<model_path>] [--out_path=<out_path>]

Options:
--pred=<pred>               The location(including filename) of the new data file in zip/csv format relative to the root 
[--model_path=<model_path>] Path(folder_name) ending with "/" of the folder where fitted models are located
[--out_path=<out_path>]     Path(folder_name) ending with "/" relative to root where to write the zip/csv with new_data and predicted target values from all 3 models named predicted_data.zip
"""

from docopt import docopt
import os
import pandas as pd
import numpy as np
from joblib import dump, load

opt = docopt(__doc__)

def main(pred, model_path="src/joblib/", out_path="results/"):
    
 
    #create datasets
    predict_data = pd.read_csv(pred)
    #checking if prediction data already contains target
    assert "unacast_session_count" not in set(predict_data.columns), "prediction data has target column"
    
    if model_path is None:
        model_path = "src/joblib/"

    #loading models
    filepath = model_path+"lgbm_model.joblib" 
    lgbm = load(filepath)

    # filepath = model_path+"xgb_model.joblib" 
    # xgb = load(filepath)

    # filepath = model_path+"catb_model.joblib" 
    # catb = load(filepath)

    # getting predictions from saved models

    y_preds_lgbm = lgbm.predict(predict_data)
    # y_preds_xgb = xgb.predict(predict_data)
    # y_preds_catb = catb.predict(predict_data)

    # capping the negative predicted values with 0
    y_preds_lgbm = list(map(lambda x: 0 if x<0 else x, y_preds_lgbm))
    # y_preds_xgb = list(map(lambda x: 0 if x<0 else x, y_preds_xgb))
    # y_preds_catb = list(map(lambda x: 0 if x<0 else x, y_preds_catb))
    
    # making session counts as integers
    y_preds_lgbm = list(map(lambda x: int(x), y_preds_lgbm))
    # y_preds_xgb = list(map(lambda x: int(x), y_preds_xgb))
    # y_preds_catb = list(map(lambda x: int(x), y_preds_catb))
    
    #adding the predicted values to dataframe

    predict_data['session_count_lightgbm'] = y_preds_lgbm
    # predict_data['session_count_xgboost'] = y_preds_xgb
    # predict_data['session_count_catboost'] = y_preds_catb
  
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
    pred_data_loc = "src/dummy_pred_data.zip"
    main(pred_data_loc)
    assert os.path.exists("results/predicted_data.zip"), "Predicted file not found in location"

if __name__ == "__main__":
    test_fun()
    main(opt["--pred"], opt["--model_path"], opt["--out_path"])
