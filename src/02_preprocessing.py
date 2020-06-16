# Author: Reiko Okamoto
# Date: 2020-06-15

"""
This script preprocesses the training and test sets. It can also be used to
preprocess new data using the imputer and one-hot encoder that were
previously fitted to the training data.

Usage: src/02_preprocessing.py --test=<test> [--train=<train>] 

Options:
--test=<test>       The location (including filename) of the test data in zip/csv format relative to the root
[--train=<train>]    The location (including filename) of the training data in zip/csv format relative to the root
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

        # save processed data
        X_train.to_csv('data/processed_train.csv', index=False)

        print('Saving preprocessed X_train successful!')

    #===================================
    # PREPROCESS X_TEST
    #===================================

    # drop observations missing `unacast_session_count`
    test_data = pd.read_csv(test)

    # create X and y
    X_test = test_data.drop('unacast_session_count', axis=1)
    y_test = test_data.loc[:, 'unacast_session_count']

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
    X_test['unacast_session_count'] = y_test

    # save processed data
    X_test.to_csv('data/processed_test.csv', index=False)

    print('Saving preprocessed X_test successful!')

    return 

def test_main():

    dummy_test_loc = 'data/dummy/dummy_test.csv'
    dummy_train_loc = 'data/dummy/dummy_train.csv'

    main(dummy_test_loc, dummy_train_loc)

    assert os.path.exists('data/preprocessed_train.csv'), 'Preprocessed training data not found in designated location'
    assert os.path.exists('data/preprocessed_test.csv'), 'Preprocessed test data not found in designated location'

opt = docopt(__doc__)

if __name__ == "__main__":
    #test_main()
    main(opt['--test'], opt['--train'])
