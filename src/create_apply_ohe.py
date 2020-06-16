import pandas as pd
import numpy as np
import pickle
from joblib import dump, load

from sklearn.preprocessing import OneHotEncoder

def create_ohe(X_train, to_encode=['income_class', 'density_class', 'climate'], filename='ohe.joblib'):
    
    """
    Return an one-hot-encoder fitted using `X_train`
    as a pickle file.
    
    Parameters
    ----------
    X_train: pd.DataFrame
        Training set
        
    to_encode: list 
        The list of the categorical variables we want to encode
        
    filename: str
        Filename to use to save encoder
    
    Returns
    -------
    bytes
    """
    ohe = OneHotEncoder(sparse=False, dtype=int)
    
    ohe.fit(X_train.loc[:, to_encode])
    
    # Save the OHE
    # pickled_ohe = pickle.dumps(ohe)
    pickled_ohe = dump(ohe, filename)
    
    return None

def apply_ohe(X, to_encode=['income_class', 'density_class', 'climate'], filename='ohe.joblib'):
    """
    Given an one-hot-encoder fit on `X_train` and
    a list of columns to encode, return a data frame.
    
    WARNING: `to_encode` must match list passed to create `ohe`

    Parameters
    ----------
    X: pd.DataFrame
        `X_train`, `X_valid` or `X_test`
    
    to_encode: list
        List of categorical variables to encode
        
    filename: str
        Filename of fitted encoder
    
    Returns
    -------
    pd.DataFrame
    
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

    # drop the columns for which we used OHE
    full_X_output = full_X_output.drop(columns=to_encode)

    #Check that the number of rows is unchanged
    assert full_X_output.shape[0] == X_output.shape[0]

    #Check that `income_class` column is not in `output_data`
    assert 'income_class' not in full_X_output.columns.to_list()

    return full_X_output