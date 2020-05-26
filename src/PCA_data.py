
#importing libraries

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_components(input_df, var_per_rqd):
    """
    This function takes the input raw data and outputs the data
    projected onto a set of orthogonal axes (i.e. principal components)
    for the provided explained variance ratio.

    Parameters
    --------------
    input_df : DataFrame,
    var_per_rqd : int, the sum of explained variance ratio of principal
        components required

    Returns
    ---------------
    DataFrame,
        with the principal components

    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_df)
    
    pca = PCA()
    pca.fit(scaled_data)

    var_ratio = pca.explained_variance_ratio_
    
    i = 0
    while i < len(var_ratio):
        if np.sum(var_ratio[0:i]) >=  var_per_rqd:
            break
        i+=1
    
    principal_components = pca.transform(scaled_data)[:, 0:i]
    return pd.DataFrame(principal_components)
    

def pca(input_df, var_per_rqd = 0.99, by_groups = False):

    """
    This function takes the entire data and performs dimensionality reduction 
    via PCA either in sections or on the entire dataframe.

    Parameters
    ---------------
    input_df : DataFrame, the entire dataframe with all the columns
    var_per_rqd : int, the sum of explained variance ratio of principal
        components required
    by_groups : boolean, wheather to perform pca on whole input data taken together
        or divide it into pre-defined categories - biba, weather, neighbourhood,
        politics and census columns and PCA on each separately
    
    Returns
    ---------------
    DataFrame,
        with the principal components
    """

    if by_groups:

        # creating categories from entire data
        df_biba = input_df.loc[:, 'monthly_number_of_sessions':'monthly_Sunday']
        
        df_neighbour = input_df.loc[:, 'longitude':'streets_per_node_proportion_4_osid']

        df_census = input_df.loc[:, 'B20004e10':'fertility_rate_2018']

        df_politics = input_df.loc[:, ['Democrats_08_Votes', 'Democrats_12_Votes' , 'Republican_08_Votes', 'Republican_12_Votes', 
                                    'Republicans_2016', 'Democrats_2016', 'Green_2016', 'Libertarians_2016']]
        
        cols_extracted = list(df_biba.columns) + list(df_neighbour.columns) + list(df_census.columns) + list(df_politics.columns)
        
        df_weather = input_df.drop(columns = cols_extracted)

        # running pca on each groups created above
        pcs_df = pd.DataFrame()
        pcs_df = pd.concat([get_components(df_biba, var_per_rqd), 
                            get_components(df_neighbour, var_per_rqd), 
                            get_components(df_census, var_per_rqd), 
                            get_components(df_politics, var_per_rqd),
                            get_components(df_weather, var_per_rqd)], axis = 1)

    else:
        pcs_df = pd.DataFrame()
        pcs_df = get_components(input_df, var_per_rqd)

    
    if input_df.shape[0] == pcs_df.shape[0]:
        return pcs_df
    else:
        print("The number of records don't match")
