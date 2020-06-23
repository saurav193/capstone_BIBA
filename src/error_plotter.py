import pandas as pd
import numpy as np

def plot_resid(model, X_train=None, y_train=None, X_valid=None, y_valid=None, plot = 'both'):
    """
    Creates scatter plot for train and validation data for a given model.
    
    Parameters
    ----------
    model: A trained model type object with a model.predict method
    
    X_train: pd.DataFrame, `X_train`
    
    y_train: Array like set of train target variable
        
    X_valid: pd.DataFrame, `X_valid`
    
    y_valid: Array like set of validation target variable
    
    plot: String, if 'valid' won't plot train, if 'train' won't plot valid, else will plot both
    
    Returns
    -------
    dictionary with the two plots, with the keys 'Train' and 'Valid' respectively
    
    """
    d = dict()
    
    if plot != 'valid':
        train_df = pd.DataFrame({'Predicted Train':model.predict(X_train), 'True Train':y_train})
        train_df['Train Error'] =  train_df['Predicted Train'] - train_df['True Train']
        train_dist = alt.Chart(train_df).mark_circle().encode(alt.X("True Train:Q"), y=alt.Y('Train Error:Q'))
        d["Train"] = train_dist
    else:
        d["Valid"] = "No Validation set given"
        
        
    if plot != 'train':
        valid_df = pd.DataFrame({'Predicted Valid':model.predict(X_valid), 'True Valid':y_valid})
        valid_df['Valid Error'] =  valid_df['Predicted Valid'] - valid_df['True Valid']
        
        valid_dist = alt.Chart(valid_df).mark_circle().encode(alt.X("True Valid:Q"), y=alt.Y('Valid Error Distance:Q'))
        d["Valid"] = valid_dist
    else:
        d["Valid"] = "No training set given""
        
    return d
