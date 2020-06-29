# Date: 2020-06-28
#
# Author: Reiko Okamoto
#
# The functions in this file can be used to report model performance.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.datasets import load_boston

def report_importance(model, n, df):
    """
    Return column names and Gini coefficients of
    n most important features.
    
    Parameters
    ----------
    model: scikit-learn estimator object 
        a fitted model with with feature_importances_ property
        
    n: int
        number of features
        
    df: pd.DataFrame
        `X_train`, `X_valid` or `X_test`
    
    Returns
    -------
    pd.DataFrame
    
    """
    # code attribution: https://tinyurl.com/ya52tn2p
    values = model.feature_importances_
    indices = (-values).argsort()[:n]
    
    # get column names of n most important features
    col_names = df.iloc[:, list(indices)].columns.to_list()
    
    # get Gini coefficient of n most important features
    gini_coeff = list(np.sort(values)[-n:][::-1])

    data = {'feature': col_names, 'Gini': gini_coeff}
    
    result = pd.DataFrame(data)
    
    return result

def test_report_importance():
    # load toy data
    X, y = load_boston(return_X_y=True)

    # split data for training
    split = train_test_split(X, y, random_state=0)
    X_train = split[0] 
    y_train = split[2]

    # fit a dummy model
    reg = GradientBoostingRegressor(random_state=0)
    reg.fit(X_train, y_train)

    result = report_importance(reg, 3, pd.DataFrame(X_train))

    assert result.shape[0] == 3, 'output should have 3 rows'
    assert result.shape[1] == 2, 'output should have 2 columns'

    # fit another dummy model
    rf = RandomForestRegressor(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)

    result = report_importance(reg, 5, pd.DataFrame(X_train))

    assert result.shape[0] == 5, 'output should have 5 rows'
    assert result.shape[1] == 2, 'output should have 2 columns'

def report_search(search):
    """
    Print the best hyperparameter settings and
    search.cv_results_ as a dataframe.
    
    Parameters
    ----------
    search: sklearn.model_selection.RandomizedSearchCV (or GridSearchCV)

    Returns
    -------
    pd.DataFrame

    """
    print(search.best_params_)
    
    results = pd.DataFrame(search.cv_results_)
    
    return results

def test_report_search():
    # load toy data
    X, y = load_boston(return_X_y=True)

    # split data for training
    split = train_test_split(X, y, random_state=0)
    X_train = split[0] 
    y_train = split[2]

    # perform a random search
    reg = GradientBoostingRegressor(random_state=0)
    param_grid = {'learning_rate': [0.04, 0.06, 0.08, 0.1], 'n_estimators': [50, 100, 150]}
    search = RandomizedSearchCV(reg, param_grid, n_iter=5, random_state=0)
    search.fit(X_train, y_train)

    result = report_search(search)

    assert isinstance(result, pd.DataFrame), 'output should be pandas.core.frame.DataFrame'
    assert 'param_learning_rate' in result.columns.tolist(), 'dataframe should contain learning_rate column'
    assert 'param_n_estimators' in result.columns.tolist(), 'dataframe should contain n_estimators column'

    # perform a grid search
    reg_02 = GradientBoostingRegressor(random_state=0)
    param_grid_02 = {'n_estimators': [50, 100, 150]}
    search_02 = GridSearchCV(reg_02, param_grid_02)
    search_02.fit(X_train, y_train)

    result_02 = report_search(search_02)
    
    assert isinstance(result_02, pd.DataFrame), 'output should be pandas.core.frame.DataFrame'
    assert 'param_n_estimators' in result_02.columns.tolist(), 'dataframe should contain n_estimators column'
    assert result_02.shape[0] == len(param_grid_02['n_estimators']), 'dataframe should have 3 rows'

def report_performance(model, X_train, y_train, X_valid, y_valid, mode='mean', floor=False):
    """
    Evaluate train and validation performance on a fitted model.
    
    Parameters
    ---------     
    model: scikit-learn estimator object
        a fitted model
    X_train: pandas.core.frame.DataFrame
        X of training set
    y_train: pandas.core.series.Series
        y of training set
    X_valid: pandas.core.frame.DataFrame        
        X of validation set
    y_valid: pandas.core.series.Series
        y of validation set     
    mode: string
        'mean' or 'median'
    floor : boolean
        if true, all the negative values are turned into 0s
    
    Returns
    -------
    errors: list
        
    """
    if mode == 'mean':
        if floor:
            errors = [np.sqrt(mean_squared_error(y_train, list(map(lambda x: 0 if x < 0 else x, model.predict(X_train))))), 
                      np.sqrt(mean_squared_error(y_valid, list(map(lambda x: 0 if x < 0 else x, model.predict(X_valid)))))]

        else:
            errors = [np.sqrt(mean_squared_error(y_train, model.predict(X_train))), 
                      np.sqrt(mean_squared_error(y_valid, model.predict(X_valid)))]
            
        print('Training RMSE:', errors[0])
        print('Validation RMSE:', errors[1])
        
    elif mode == 'median':
        if floor:
            errors = [mean_absolute_error(y_train, list(map(lambda x: 0 if x < 0 else x, model.predict(X_train)))), 
                      mean_absolute_error(y_valid, list(map(lambda x: 0 if x < 0 else x, model.predict(X_valid))))]

        else:
            errors = [mean_absolute_error(y_train, model.predict(X_train)), 
                      mean_absolute_error(y_valid, model.predict(X_valid))]
            
        print('Training MAE:', errors[0])
        print('Validation MAE:', errors[1])
        
    return errors

def test_report_performance():
    
    # load toy data
    X, y = load_boston(return_X_y=True)

    # split data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit a model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    result_mean = report_performance(lr, X_train, y_train, X_test, y_test)
    assert len(result_mean) == 2, 'returned list should be of length 2'
    assert (result_mean[0] > 0) & (result_mean[1] > 0), 'both values should be greater than 0'

    result_mean_fl = report_performance(lr, X_train, y_train, X_test, y_test, floor=True)
    assert len(result_mean_fl) == 2, 'returned list should be of length 2'
    assert (result_mean_fl[0] > 0) & (result_mean_fl[1] > 0), 'both values should be greater than 0'

    result_median = report_performance(lr, X_train, y_train, X_test, y_test, mode='median')
    assert len(result_median) == 2, 'returned list should be of length 2'

    result_median_fl = report_performance(lr, X_train, y_train, X_test, y_test, mode='median', floor=True)
    assert len(result_median_fl) == 2, 'returned list should be of length 2'
    assert (result_median_fl[0] > 0) & (result_median_fl[1] > 0), 'both values should be greater than 0'
