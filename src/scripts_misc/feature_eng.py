import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

def comb_cols(input_df):

    """
    Given the entire dataframe (all columns), combine some columns
    into new features using domain knowledge.

    Parameters
    ----------
    input_df : pandas.core.frame.DataFrame
       the entire dataframe that has been imputed
        
    Returns
    -------
    pandas.core.frame.DataFrame
        with new features combining some existing features
    """
    output_df = input_df.copy()
    
    # creating list of equipments to group together
    
    equipments = ['slide', 'climb', 'tube', 'overhang', 'bridge', 'swing', 'obsta', 'crawls']
    
    # creating dictionary of lists of columns to combine: 2 (monthly and historic) lists for each type of equipment
    new_cols_list = {}
    cols_to_drop = []
    cols_added = 0 # counting added cols by combining other cols
    
    for equipment in equipments:
        new_cols_list["monthly_"+equipment] = [i for i in input_df.columns if re.match('monthly_.*'+equipment+'.*', i) and 
                                                                                not (re.match('monthly_.*'+equipment+'.*tube', i))]
        new_cols_list["historic_"+equipment] = [i for i in input_df.columns if re.match('historic_.*'+equipment+'.*', i) and 
                                                                                not (re.match('historic_.*'+equipment+'.*tube', i))]
        if new_cols_list["monthly_"+equipment]==[]:
            new_cols_list.pop("monthly_"+equipment)
            
    for key, val in new_cols_list.items():
        output_df[key+"_count_comb"] = np.sum(output_df.loc[:, val], axis = 1)
        cols_to_drop += val # add old columns to a list of columns to drop
        cols_added += 1 
      
    # grouping together 'monthly_hour_*'' between 10 pm and 7 am
    monthly_hour_night = input_df.loc[:, 'monthly_hour_0':'monthly_hour_6'].columns.to_list() \
                         + input_df.loc[:, ['monthly_hour_22','monthly_hour_23']].columns.to_list() 
    
    output_df['monthly_hour_night'] = np.sum(input_df.loc[:, monthly_hour_night], axis=1)
    cols_added += 1
    
    # adding old 'monthly_hour_*' columns to list to drop
    cols_to_drop += monthly_hour_night
        
    # grouping together `historic_hour_*` between 10 pm and 7 am
    historic_hour_night = input_df.loc[:, 'historic_hour_0':'historic_hour_6'].columns.to_list() \
                         + input_df.loc[:, ['historic_hour_22','historic_hour_23']].columns.to_list() 
    
    output_df['historic_hour_night'] = np.sum(input_df.loc[:, historic_hour_night], axis=1)
    cols_added += 1
    
    # adding old 'historic_hour_*' columns to list to drop
    cols_to_drop += historic_hour_night

    # combining wind speed columns
    output_df['avg_wind_calm'] = input_df['avg_wind_0_1']
    output_df['avg_wind_light_air'] = np.sum(input_df.loc[:, ['avg_wind_1_2','avg_wind_2_3','avg_wind_3_4']], axis = 1)
    output_df['avg_wind_light_br'] = np.sum(input_df.loc[:, ['avg_wind_4_5','avg_wind_5_6','avg_wind_6_7','avg_wind_7_8']], axis = 1)
    output_df['avg_wind_gentle_br'] = np.sum(input_df.loc[:, ['avg_wind_8_9','avg_wind_9_10','avg_wind_10_11','avg_wind_11_12']], axis = 1)
    output_df['avg_wind_moderate_br'] = input_df['avg_wind_12_above']
    cols_added += 5

    output_df['monthly_ws_calm'] = input_df['monthly_ws_below_2']
    output_df['monthly_ws_light_air'] = input_df['monthly_ws_2_to_4']
    output_df['monthly_ws_light_br'] = np.sum(input_df.loc[:, ['monthly_ws_4_to_6','monthly_ws_6_to_8']], axis = 1)
    output_df['monthly_ws_gentle_br'] = np.sum(input_df.loc[:, ['monthly_ws_8_to_10','monthly_ws_10_to_12']], axis = 1)
    output_df['monthly_ws_moderate_br'] = np.sum(input_df.loc[:, ['monthly_ws_12_to_14','monthly_ws_14_to_16','monthly_ws_above_16']], axis = 1)
    cols_added += 5

    output_df['historic_ws_calm'] = input_df['historic_ws_below_2']
    output_df['historic_ws_light_air'] = input_df['historic_ws_2_to_4']
    output_df['historic_ws_light_br'] = np.sum(input_df.loc[:, ['historic_ws_4_to_6','historic_ws_6_to_8']], axis = 1)
    output_df['historic_ws_gentle_br'] = np.sum(input_df.loc[:, ['historic_ws_8_to_10','historic_ws_10_to_12']], axis = 1)
    output_df['historic_ws_moderate_br'] = np.sum(input_df.loc[:, ['historic_ws_12_to_14','historic_ws_14_to_16','historic_ws_above_16']], axis = 1)
    cols_added += 5
    
    # dropping old wind speed columns
    cols_to_drop += ['avg_wind_0_1','avg_wind_1_2','avg_wind_2_3','avg_wind_3_4','avg_wind_4_5',
                    'avg_wind_5_6','avg_wind_6_7','avg_wind_7_8','avg_wind_8_9',
                    'avg_wind_9_10','avg_wind_10_11','avg_wind_11_12','avg_wind_12_above',
                    'monthly_ws_below_2','monthly_ws_2_to_4','monthly_ws_4_to_6','monthly_ws_6_to_8',
                    'monthly_ws_8_to_10','monthly_ws_10_to_12','monthly_ws_12_to_14','monthly_ws_14_to_16',
                    'monthly_ws_above_16','historic_ws_below_2','historic_ws_2_to_4','historic_ws_4_to_6',
                    'historic_ws_6_to_8','historic_ws_8_to_10','historic_ws_10_to_12','historic_ws_12_to_14',
                    'historic_ws_14_to_16','historic_ws_above_16']
    
    # averaging fertility rate columns
    output_df['avg_fertility_rate'] = np.mean(input_df.loc[:, 'fertility_rate_2003':'fertility_rate_2018'], axis=1)
    cols_to_drop += input_df.loc[:, 'fertility_rate_2003':'fertility_rate_2018'].columns.to_list()
    cols_added += 1

    # dropping other columns that's been grouped together
    output_df = output_df.drop(columns = cols_to_drop)   

    # checking that the correct number of columns have been introduced to and dropped from the input data
    assert len(input_df.columns) + cols_added - len(cols_to_drop) == len(output_df.columns)             
    
    # checking that number of rows should be same
    assert input_df.shape[0] == output_df.shape[0]
   
    return output_df

    
def clean_categorical(X_train, X_valid, to_encode=['income_class', 'density_class', 'climate']):
    """
    Fits one-hot encoder on categorical variables using X_train, 
    and transform X_train and X_valid
    
    Parameters
    ----------
    X_train : pandas.core.frame.DataFrame
    X_valid : pandas.core.frame.DataFrame
    to_encode : list
        The list of the categorical variables we want to encode
    
    Returns
    -------
    (X_train_output, X_valid_output) : tuple of pandas.core.frame.DataFrame
    
    """
    X_train_output = X_train.copy()
    X_valid_output = X_valid.copy()

    # apply One-Hot-Encoding to each one of the categorical variable
    
    ohe = OneHotEncoder(sparse=False, dtype=int)
    
    sub_X_train = ohe.fit_transform(X_train_output.loc[:, to_encode])
    sub_X_valid = ohe.transform(X_valid_output.loc[:, to_encode])
    
    # get names of encoded columns
    ohe_cols = np.concatenate(ohe.categories_).ravel()
    
    # create data frames containing encoded columns (preserve old row indices)
    sub_df_train = pd.DataFrame(sub_X_train, index=X_train.index, columns=ohe_cols)
    sub_df_valid = pd.DataFrame(sub_X_valid, index=X_valid.index, columns=ohe_cols)
    
    # concatenate with existing data frame
    X_train_output = pd.concat((X_train_output, sub_df_train), axis=1)
    X_valid_output = pd.concat((X_valid_output, sub_df_valid), axis=1)

    # drop the columns for which we used OHE
    X_train_output = X_train_output.drop(columns=to_encode)
    X_valid_output = X_valid_output.drop(columns=to_encode)

    #Check that the number of rows is unchanged

    assert X_train.shape[0] == X_train_output.shape[0]
    assert X_valid.shape[0] == X_valid_output.shape[0]

    return (X_train_output, X_valid_output)
