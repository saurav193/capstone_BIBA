
#importing libraries

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

principal_components = []
num_components = []

def get_pca_trans_data(input_df, var_per_rqd):
    """
    This function takes the input raw data and outputs the data
    projected onto a set of orthogonal axes (i.e. principal components)
    for the provided explained variance ratio.

    Parameters
    --------------
    input_df : pandas.DataFrame
    var_per_rqd : float
        percentage of variance explained by all the selected components (e.g. 0.95)

    Returns
    ---------------
    pandas.DataFrame,
        data projected onto orthogonal axes with reduced dimensionality, if attainable

    """
    global principal_components
    global num_components

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
    
    pca_trans_data = pca.transform(scaled_data)[:, 0:i]
    
    #storing the principal components for transforming the validation data
    principal_components.append(pca.components_)

    #storing the # components for given percentage explained variance
    num_components.append(i)

    return pd.DataFrame(pca_trans_data)
    

def pca_fit_transform(input_df, var_per_rqd = 0.99, by_groups = False):

    """
    This function takes the entire data and performs dimensionality reduction 
    via PCA either in sections or on the entire dataframe.

    Parameters
    ---------------
    input_df : pandas.DataFrame
       the entire dataframe with all the columns that has been preprocessed using preprocessing.py
    var_per_rqd : float
       percentage of variance explained by all the selected components (e.g. 0.95)
    by_groups : bool
       if True, divide the input data into pre-defined categories (e.g. Biba, weather, neighbourhood,
       politics, census) and perform PCA separately. Otherwise, perform PCA on the entire dataset.
    
    Returns
    ---------------
    pandas.DataFrame
        data projected onto orthogonal axes with reduced dimensionality, if attainable
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
        pcs_df = pd.concat([get_pca_trans_data(df_biba, var_per_rqd), 
                            get_pca_trans_data(df_neighbour, var_per_rqd), 
                            get_pca_trans_data(df_census, var_per_rqd), 
                            get_pca_trans_data(df_politics, var_per_rqd),
                            get_pca_trans_data(df_weather, var_per_rqd)], axis = 1)

    else:
        pcs_df = pd.DataFrame()
        pcs_df = get_pca_trans_data(input_df, var_per_rqd)

    
    if input_df.shape[0] == pcs_df.shape[0]:
        return pcs_df
    else:
        print("The number of records don't match")

def pca_transform(input_df, var_per_rqd = 0.99, by_groups = False):

    """
    This function transforms the entire data into reduced dimensions using 
    principal components computed previously in pca_fit_transform(). 
    The data is transformed either in sections or on the entire dataframe.

    Parameters
    ---------------
    input_df : pandas.DataFrame
       the dataframe with all the columns that has been preprocessed using preprocessing.py
    var_per_rqd : float
       percentage of variance explained by all the selected components (e.g. 0.95)
    by_groups : bool
       if True, divide the input data into pre-defined categories (e.g. Biba, weather, neighbourhood,
       politics, census) and perform PCA separately. Otherwise, perform PCA on the entire dataset.
    
    Returns
    ---------------
    pandas.DataFrame
        data projected onto orthogonal axes with reduced dimensionality, if attainable
    """
    global principal_components
    global num_components

    if by_groups:

        # creating categories from entire data
        df_biba = input_df.loc[:, 'monthly_number_of_sessions':'monthly_Sunday']
        
        df_neighbour = input_df.loc[:, 'longitude':'streets_per_node_proportion_4_osid']

        df_census = input_df.loc[:, 'B20004e10':'fertility_rate_2018']

        df_politics = input_df.loc[:, ['Democrats_08_Votes', 'Democrats_12_Votes' , 'Republican_08_Votes', 'Republican_12_Votes', 
                                    'Republicans_2016', 'Democrats_2016', 'Green_2016', 'Libertarians_2016']]
        
        cols_extracted = list(df_biba.columns) + list(df_neighbour.columns) + list(df_census.columns) + list(df_politics.columns)
        
        df_weather = input_df.drop(columns = cols_extracted)

        # transforming each group of data created above by matrix multiplication with pca components from train data
        # taking only the number of columns in train data for each group
        biba_trans_data = (df_biba.to_numpy() @ principal_components[0].T)[:, 0:num_components[0]]
        neighbour_trans_data = (df_neighbour.to_numpy() @ principal_components[1].T)[:, 0:num_components[1]]
        census_trans_data = (df_census.to_numpy() @ principal_components[2].T)[:, 0:num_components[2]]
        politics_trans_data =  (df_politics.to_numpy() @ principal_components[3].T)[:, 0:num_components[3]]
        weather_trans_data = (df_weather.to_numpy() @ principal_components[4].T)[:, 0:num_components[4]]

        pcs_df = pd.DataFrame()
        pcs_df = pd.concat([pd.DataFrame(biba_trans_data), 
                            pd.DataFrame(neighbour_trans_data), 
                            pd.DataFrame(census_trans_data), 
                            pd.DataFrame(politics_trans_data),
                            pd.DataFrame(weather_trans_data)], axis = 1)

    else:
        # transfomring the entire data by matrix multiplication with pca components from train data
        # taking only the number of columns in train data for each group
        transformed_data = (input_df.to_numpy() @ principal_components[0].T)[:, 0:num_components[0]]
        pcs_df = pd.DataFrame(transformed_data)

    if input_df.shape[0] == pcs_df.shape[0]:
        principal_components = []
        num_components = []
        return pcs_df
    else:
        print("The number of records don't match")
