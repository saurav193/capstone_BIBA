import pandas as pd
import numpy as np
import altair as alt

def plot_resid(model, X_train=None, y_train=None, X_valid=None, y_valid=None, plot = 'both'):
    """
    Creates scatter plot plus smooth line for train and validation data for a given model.
    
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
        train_df = pd.DataFrame({'Predicted Train':list(map(lambda x: 0 if x<0 else x, model.predict(X_train))), 'True train':y_train}).sort_values(by=['True train'])
        train_df['Train Error'] =  train_df['Predicted Train'] - train_df['True train']
        train_dist_point = alt.Chart(train_df).mark_circle().encode(alt.X("True train:Q"), y=alt.Y('Train Error:Q', axis=alt.Axis(title='Train error (rolling_mean)')))
        train_dist_line = alt.Chart(train_df).mark_line(color='red',size=1).transform_window(rolling_mean='mean(Train Error)',frame=np.array([-20, 20])
                                                                                        ).encode(alt.X("True train:Q"), y=alt.Y('rolling_mean:Q'))
        train_dist = train_dist_point + train_dist_line
        d["Train"] = train_dist
    else:
        d["Valid"] = "No Validation set given"
        
        
    if plot != 'train':
        valid_df = pd.DataFrame({'Predicted Valid':list(map(lambda x: 0 if x<0 else x, model.predict(X_valid))), 'True Valid':y_valid}).sort_values(by=['True Valid'])
        valid_df['Valid Error'] =  valid_df['Predicted Valid'] - valid_df['True Valid']
        valid_dist_point = alt.Chart(valid_df).mark_circle().encode(alt.X("True Valid:Q", axis=alt.Axis(title='True test')), y=alt.Y('Valid Error:Q', axis=alt.Axis(title='Test error (rolling_mean)')))
        valid_dist_line = alt.Chart(valid_df).mark_line(color='red',size=1).transform_window(rolling_mean='mean(Valid Error)',frame=np.array([-40, 40])
                                                                                        ).encode(alt.X("True Valid:Q"), y=alt.Y('rolling_mean:Q'))
        valid_dist = valid_dist_point + valid_dist_line
        d["Valid"] = valid_dist
    else:
        d["Valid"] = "No training set given"
        
    return d
