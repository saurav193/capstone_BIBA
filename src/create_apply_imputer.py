import pandas as pd
import numpy as np
import re
import pickle
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def create_imputer(X_train, filename='imputer.joblib'):

    """
    Fit a transformer on `X_train` and save it in
    .joblib format
    
    Parameters
    ----------
    X_train: pd.DataFrame
        Training set
        
    filename: str
        Filename to use to save transformer
    
    """
    
    #======================================
    # LOAD DATA FRAME
    #======================================

    # df = pd.read_csv('../data/train_data.zip')
    df = X_train

    #======================================
    # IDENTIFY COLUMNS TO IMPUTE
    #======================================

    # Impute with 0
    monthly_count_equipment = df.loc[:, 'monthly_count_slide_single':'monthly_count_climber'].columns.to_list()
    historic_session = df.loc[:, 'historic_number_of_sessions':'historic_avg_mod_plus_vig'].columns.to_list()
    historic_hour = df.loc[:, 'historic_hour_0':'historic_hour_23'].columns.to_list()
    historic_count_equipment = df.loc[:, 'historic_count_bridge':'historic_count_zipline'].columns.to_list()
    historic_weather = df.loc[:, 'historic_cloudy':'historic_snow'].columns.to_list()
    OSM = df.loc[:, 'n': 'streets_per_node_proportion_7_osid'].columns.to_list()
    zero_misc = ['days_since_first_sess', 'perfect_days', 'Green_2016', 'Number_of_holidays']

    zero_imp_features = monthly_count_equipment + historic_session + historic_hour \
                        + historic_count_equipment + historic_weather + OSM + zero_misc

    # Impute with mean
    weather = df.loc[:, 'weather_clear':'avg_wind_12_above'].columns.to_list()
    mean_misc = ['walk_score', 'bike_score', 'Poor_physical_health_days', 'Poor_mental_health_days', 'Adult_smoking']

    mean_imp_features = weather + mean_misc

    #======================================
    # CREATE TRANSFORMERS
    #======================================

    # Create transformer for 0 imputation
    zero_transformer = SimpleImputer(strategy='constant', fill_value=0)

    # Create transformer for mean imputation
    mean_transformer = SimpleImputer(strategy='mean')

    # Create transformer for `Republicans_08_Votes`
    rep_08_votes_transformer = SimpleImputer(strategy='constant', fill_value=193841)

    # Create transformer for `Democrats_08_Votes`
    dem_08_votes_transformer = SimpleImputer(strategy='constant', fill_value=123594)

    # Create transformer for `Republican_12_Votes`
    rep_12_votes_transformer = SimpleImputer(strategy='constant', fill_value=164676)

    # Create transformer for `Democrats_12_Votes`
    dem_12_votes_transformer = SimpleImputer(strategy='constant', fill_value=122640)

    # Create transformer for `Republicans_2016`
    rep_2016_transformer = SimpleImputer(strategy='constant', fill_value=163387)

    # Create transformer for `Democrats_2016`
    dem_2016_transformer = SimpleImputer(strategy='constant', fill_value=116454)

    # Create transformer for `Libertarians_2016`
    lib_2016_transformer = SimpleImputer(strategy='constant', fill_value=18725)

    #======================================
    # PUT IT ALL TOGETHER
    #======================================

    imputer = ColumnTransformer(
        transformers=[
            ('zero', zero_transformer, zero_imp_features),
            ('mean', mean_transformer, mean_imp_features),
            ('rep_08_votes', rep_08_votes_transformer, ['Republican_08_Votes']),
            ('dem_08_votes', dem_08_votes_transformer, ['Democrats_08_Votes']),
            ('rep_12_votes', rep_12_votes_transformer, ['Republican_12_Votes']),
            ('dem_12_votes', dem_12_votes_transformer, ['Democrats_12_Votes']),
            ('rep_2016', rep_2016_transformer, ['Republicans_2016']),
            ('dem_2016', dem_2016_transformer, ['Democrats_2016']),
            ('lib_2016', lib_2016_transformer, ['Libertarians_2016'])
        ],
        remainder='passthrough'
    ) 

    # Check that unspecified columns are passed through
    assert imputer.remainder == 'passthrough'
        
    # Check that the output is comprised of 9 transformers
    assert len(imputer.transformers) == 9

    #======================================
    # SAVE IMPUTER FOR FUTURE USE
    #======================================
    
    # Fit the column transformer on X_train
    imputer = imputer.fit(X_train)

    # Save the imputer
    # pickled = pickle.dumps(imputer)
    pickled_imputer = dump(imputer, filename)
    
    return None


def apply_imputer(X, filename='imputer.joblib'):
    """
    Load a transformer fit on `X_train`.
    Return the imputed dataframe.

    Parameters
    ----------
    X: pd.DataFrame
        `X_train`, `X_valid` or `X_test`
    
    filename: str
        Filename of fitted imputer 
    
    Returns
    -------
    pd.DataFrame
    
    """
    
    # load in transformer that's fit on `X_train`
    imputer = load(filename)
    
    # Transform data frame accordingly
    imputed_X = imputer.transform(X)
        
    cols = []
    
    # Grab column names of imputed features
    for i in range(len(imputer.transformers_) - 1):
        cols += imputer.transformers_[i][2]
    
    # Grab column names of features that were passed through unchanged
    cols += [X.columns[i] for i in imputer.transformers_[-1][2]]
    
    # Grab old order of columns
    old_cols = X.columns.to_list()
    
    # Reshuffle column order of new dataframes to match old one
    imputed_X = pd.DataFrame(imputed_X, index=X.index, columns=cols).reindex(columns=old_cols)
    
    # Cast each pandas object to its previous dtype
    types = X.dtypes.to_dict()
    imputed_X = imputed_X.astype(types)
    
    # Check that the number of rows is unchanged
    assert imputed_X.shape[0] == X.shape[0]
    
    # Check that the first column of `X_train` is `external_id`
    assert imputed_X.columns[0] == 'external_id'
    
    return imputed_X