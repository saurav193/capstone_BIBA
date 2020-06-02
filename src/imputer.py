import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def create_imputer(X_train):

    """
    Fit all transformers using `X_train`.
    
    Parameters
    ----------
    X_train: pd.DataFrame
        Training set
    
    Returns
    -------
    sklearn.compose._column_transformer.ColumnTransformer
    
    """
    
    #======================================
    # IMPORT DATA FRAME
    #======================================

    df = pd.read_csv('../data/train_data.zip')

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
    # PUTTING IT ALL TOGETHER
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
    
    return imputer

def impute_data(X_train, X_valid):
    """
    Given a transformer fit on `X_train`, return the imputed dataframes.
    
    Note: add code later if you want to impute `X_test`
    
    Parameters
    ----------
    imputer: sklearn.compose._column_transformer.ColumnTransformer
        imputer
    
    X_train: pd.DataFrame
        `X_train`
        
    X_valid: pd.DataFrame
        `X_valid`
    
    Returns
    -------
    tuple
        where imputed_dfs[0] is imputed `X_train` and
        imputed_df[1] is imputed `X_valid`
    """
    
    imputer = create_imputer(X_train)
    
    imp_X_train = imputer.fit_transform(X_train)
    imp_X_valid = imputer.transform(X_valid)
        
    cols = []
    
    # Grab column names of imputed features
    for i in range(len(imputer.transformers_) - 1):
        cols += imputer.transformers_[i][2]
    
    # Grab column names of features that were passed through unchanged
    cols += [X_train.columns[i] for i in imputer.transformers_[-1][2]]
    
    # Grab old order of columns
    old_cols = X_train.columns.to_list()
    
    # Create new dataframes
    # Reshuffle column order of new dataframes to match old one
    imp_X_train = pd.DataFrame(imp_X_train, columns=cols).reindex(columns=old_cols)
    imp_X_valid = pd.DataFrame(imp_X_valid, columns=cols).reindex(columns=old_cols)
    
    # Cast each pandas object to its previous dtype
    types = X_train.dtypes.to_dict()
    
    imp_X_train = imp_X_train.astype(types)
    imp_X_valid = imp_X_valid.astype(types)
    
    imputed_dfs = (imp_X_train, imp_X_valid)
    
    # Check that the number of rows is unchanged in `X_train`
    assert imputed_dfs[0].shape[0] == X_train.shape[0]
    
    # Check that the first column of `X_train` is `external_id`
    assert imputed_dfs[0].columns[0] == 'external_id'
    
    # Check that the number of rows is unchanged in `X_valid`
    assert imputed_dfs[1].shape[0] == X_valid.shape[0]
    
    # Check that the first column of `X_valid` is `external_id`
    assert imputed_dfs[1].columns[0] == 'external_id'
    
    return imputed_dfs
    