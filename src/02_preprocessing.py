# Author: Reiko Okamoto
# Date: 2020-06-15

"""
This script performs preprocessing. 

Usage: src/02_preprocessing.py --train=<train> --test=<test>

Options:
--train=<train>            a path pointing to the training set (validation included)
--test=<test>              a path pointing to the test set

"""
import pandas as pd
import numpy as np

from docopt import docopt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from drop import *
from create_apply_imputer import *
from feature_eng import *
from create_apply_ohe import *

def main(train, test):
    """
    
    Parameters
    ----------
    train: str
        path to training set

    test: str
        path to test set

    output: str
        output directory
    
    """
    #===================================
    # PREPROCESS X_TRAIN_VALID
    #===================================

    train_data = pd.read_csv(train)

    # Deal with missing `unacast_session_count`
    train_data = drop_missing_unacast(train_data)

    # Create X and y
    X_train = train_data.drop('unacast_session_count', axis=1)
    y_train = train_data.loc[:, 'unacast_session_count']

    # Fit and save imputer
    create_imputer(X_train, filename='src/joblib/imputer.joblib')

    # Transform data using imputer
    X_train = apply_imputer(X_train, filename='src/joblib/imputer.joblib')

    # Perform feature engineering
    X_train = comb_cols(X_train)

    # Perform feature selection
    X_train = drop_columns(X_train)

    # Fit and save one-hot encoder
    create_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

    # Transform data using one-hot encoder
    X_train = apply_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

    # Attach y
    X_train['unacast_session_count'] = y_train

    # Save processed data
    X_train.to_csv('data/processed_train.csv', index=False)

    #===================================
    # PREPROCESS X_TEST
    #===================================

    # Deal with missing `unacast_session_count`
    test_data = pd.read_csv(test)

    # Create X and y
    X_test = test_data.drop('unacast_session_count', axis=1)
    y_test = test_data.loc[:, 'unacast_session_count']

    # Transform data using saved imputer
    X_test = apply_imputer(X_test, filename='src/joblib/imputer.joblib')

    # Perform feature engineering
    X_test = comb_cols(X_test)

    # Perform feature selection
    X_test = drop_columns(X_test)

    # Transfrom data using saved one-hot encoder
    X_test = apply_ohe(X_test, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

    # Attach y
    X_test['unacast_session_count'] = y_test

    # Save processed data
    X_test.to_csv('data/processed_test.csv', index=False)

    return 

opt = docopt(__doc__)

if __name__ == "__main__":
    main(opt['--train'], opt['--test'])