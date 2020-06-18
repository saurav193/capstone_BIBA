import pandas as pd
import numpy as np
import pickle
from joblib import dump, load

from sklearn.preprocessing import OneHotEncoder

def create_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='ohe.joblib'):
    
    """
    Return an one-hot encoder fitted using `X_train`
    as a joblib file.
    
    Parameters
    ----------
    X_train: pandas.core.frame.DataFrame
        Training set
        
    to_encode: list 
        The list of the categorical variables we want to encode
        
    filename: str
        Filename to use to save encoder
    
    """
    ohe = OneHotEncoder(sparse=False, dtype=int)
    
    ohe.fit(X_train.loc[:, to_encode])
    
    # check number of columns to encode
    assert len(ohe.categories_) == len(to_encode)

    # Save the OHE
    dump(ohe, filename)
    
    return 

def apply_ohe(X, to_encode=['income_class', 'density_class', 'climate'], filename='ohe.joblib'):
    """
    Given an one-hot encoder fit on `X_train` and
    a list of columns to encode, return a data frame
    with the one-hot encoder applied.
    
    WARNING: `to_encode` must match list passed to create the encoder

    Parameters
    ----------
    X: pandas.core.frame.DataFrame
        `X_train`, `X_valid` or `X_test`
    
    to_encode: list
        List of categorical variables to encode
        
    filename: str
        Filename of fitted encoder
    
    Returns
    -------
    pandas.core.frame.DataFrame
    
    """
    X_output = X.copy()
    
    ohe = load(filename)
    sub_X_output = ohe.transform(X_output.loc[:, to_encode])
    
    # get names of encoded columns
    ohe_cols = np.concatenate(ohe.categories_).ravel()
    
    # create data frames containing encoded columns (preserve old row indices)
    sub_X_output = pd.DataFrame(sub_X_output, index=X.index, columns=ohe_cols)
    
    # concatenate with existing data frame
    full_X_output = pd.concat((X_output, sub_X_output), axis=1)

    # drop the columns to which the encoder was applied
    full_X_output = full_X_output.drop(columns=to_encode)

    # check that the number of rows is unchanged
    assert full_X_output.shape[0] == X_output.shape[0]

    # check that first column in `to_encode` is not in `output_data`
    assert to_encode[0] not in full_X_output.columns.to_list()

    return full_X_output