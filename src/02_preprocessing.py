# Author: Reiko Okamoto
# Date: 2020-06-15

"""
This script preprocesses and saves the train and test sets as zip files. It can also
preprocess new data using the imputer and one-hot encoder that were
previously fitted to the training data. It also saves dummy train and test
sets as zip files that can be used to test functions.

Usage: src/02_preprocessing.py --test=<test> [--train=<train>] 

Options:
--test=<test>       The location (including filename) of the test data in zip format relative to the root
[--train=<train>]    The location (including filename) of the training data in zip format relative to the root
"""
from docopt import docopt
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from drop import drop_columns, drop_missing_unacast
from create_apply_imputer import create_imputer, apply_imputer
from feature_eng import comb_cols
from create_apply_ohe import create_ohe, apply_ohe

def main(test, train=None):
    
    compression_opts = dict(method='zip', archive_name='out.csv')  

    #===================================
    # PREPROCESS X_TRAIN_VALID
    #===================================

    if train is not None:

        train_data = pd.read_csv(train)

        # drop observations missing `unacast_session_count`
        train_data = drop_missing_unacast(train_data)

        # create X and y
        X_train = train_data.drop('unacast_session_count', axis=1)
        y_train = train_data.loc[:, 'unacast_session_count']

        # fit and save imputer
        create_imputer(X_train, filename='src/joblib/imputer.joblib')

        # transform data using imputer
        X_train = apply_imputer(X_train, filename='src/joblib/imputer.joblib')

        # perform feature engineering
        X_train = comb_cols(X_train)

        # perform feature selection
        X_train = drop_columns(X_train)

        # fit and save one-hot encoder
        create_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

        # transform data using one-hot encoder
        X_train = apply_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

        print('Preprocessing X_train successful!')

        # attach y
        X_train['unacast_session_count'] = y_train

        # save preprocessed data
        X_train.to_csv('data/processed_train.zip', index=False, compression=compression_opts)

        # save preprocessed dummy data (first 100 rows)
        X_train.head(100).to_csv('data/dummy/dummy_train_data.zip', index=False, compression=compression_opts)

        print('Saving preprocessed X_train successful!')

    #===================================
    # PREPROCESS X_TEST
    #===================================

    # drop observations missing `unacast_session_count`
    test_data = pd.read_csv(test)
    if 'unacast_session_count' in list(test_data.columns):
        test_data = drop_missing_unacast(test_data)

    # create X and y
    if 'unacast_session_count' in list(test_data.columns):
        X_test = test_data.drop('unacast_session_count', axis=1)
        y_test = test_data.loc[:, 'unacast_session_count']
    else:
        X_test = test_data

    # transform data using saved imputer
    X_test = apply_imputer(X_test, filename='src/joblib/imputer.joblib')

    # perform feature engineering
    X_test = comb_cols(X_test)

    # perform feature selection
    X_test = drop_columns(X_test)

    # transfrom data using saved one-hot encoder
    X_test = apply_ohe(X_test, to_encode=['income_class', 'density_class', 'climate'], filename='src/joblib/ohe.joblib')

    print('Preprocessing X_test successful!')

    # attach y
    if 'unacast_session_count' in list(test_data.columns):
        X_test['unacast_session_count'] = y_test

    # save processed data
    if train is not None: 
        X_test.to_csv('data/processed_test.zip', index=False, compression=compression_opts)
        # save preprocessed dummy data (first 20 rows)
        X_test.head(20).to_csv('data/dummy/dummy_test_data.zip', index=False, compression=compression_opts)
        print('Saving preprocessed X_test successful!')
    else:
        X_test.to_csv('data/processed_pred.zip', index=False, compression=compression_opts)
        # save preprocessed dummy data (first 20 rows)
        X_test.head(20).to_csv('data/dummy/dummy_pred_data.zip', index=False, compression=compression_opts)
        print('Saving preprocessed X_pred successful!')

    return

def test_main():
    assert os.path.exists('data/processed_test.zip'), 'Processed test set not found in designated location'
    assert os.path.exists('data/dummy/dummy_test_data.zip'), 'Processed dummy test set not found in designated location'

opt = docopt(__doc__)

if __name__ == "__main__":
    main(opt['--test'], opt['--train'])
    test_main()
