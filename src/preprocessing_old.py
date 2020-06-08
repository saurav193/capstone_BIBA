#Download libraries
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# The two main functions are `preprocessing_na()` do impute NaNs 
# and delete some columns with a very low fill rate, 
# `clean_categorical()` to use OHE on categorical variables.


def dict_to_columns_df(col, key, val):
    """
    This function takes a dataframe column which is in the
    form of list of dictionaries and creates a dataframe
    from the keys of the in the inner list of dictionaries 
    e.g. "[{'key': A, 'val': 1}, {'key': B, 'val': 2}]"
    
    Parameters
    ----------------
    col : DataFrame Series, the columns whose values are the in the format
    of a list of dictionaries.
    
    key : the keys in the inner dictionary from which column names are to be extracted
    
    val : the keys in the inner dictionary from which values in the column needs to
    be extracted
    
    
    Returns
    ----------------
    DataFrame
        With the new columns created from the keys of the inner dictionary
        
    """
    key_list = set()
    i=0
    # getting all the new column names
    while i < len(col):
        if type(col[i]) != float:
            dic_list = eval(col[i]) #converting col value from string to list
            for dic in range(len(dic_list)):
                if re.match('[a-zA-Z]', dic_list[dic][str(key)][0]): #removing spanish names
                    key_list.add("monthly_"+dic_list[dic][str(key)])
        i+=1
    
    all_cols_dict = defaultdict(list)
    
    i = 0
    while i < len(col):
        if type(col[i]) != float:
            dic_list = eval(col[i]) #converting col value from string to list

            for col_names in list(key_list):
                flag = 0 #to check if a column name exists in the dictionary
                for dic in range(len(dic_list)):
                    if dic_list[dic][str(key)] == col_names[8:]: #getting values from the inner dictionary matching the key
                        all_cols_dict[col_names].append(dic_list[dic][str(val)]) #putting inner dict values to new default dict
                        flag = 1
                        break
                
                if flag==0:
                    all_cols_dict[col_names].append(None)

        else:
            for col_names in list(key_list):
                all_cols_dict[col_names].append(None)

        i+=1
    new_cols_df = pd.DataFrame(all_cols_dict)
    
    # checking new df has same number of columns as given column
    if new_cols_df.shape[0] == col.shape[0]:
        return new_cols_df
    else:
        print("Column dimensions don't match")
        
def biba_pp(full_data):  
    
    """
    Performs the pre-processing of the columns for the biba data
    
    Parameters
    ---------------
    
    full_data : DataFrame, with no operations done on the biba columns
    
    Returns
    ---------------
    DataFrame
        with processed biba columns
    
    """
    biba_games_df = pd.DataFrame()
    biba_games_df = pd.concat([full_data.loc[:, 'monthly_number_of_sessions':'distance_to_nearest_bus_stop'],
                               full_data.loc[:, 'days_since_first_sess':'historic_snow']], axis = 1)
                               
    #extracting categorical features
    categorical_features = biba_games_df.loc[:, biba_games_df.dtypes == "object"]
     
    # creating cols from list of dictionaries
    monthly_survey_df = dict_to_columns_df(categorical_features['monthly_survey'], 'question', 'avg_answer')
    monthly_weekday_counts_df = dict_to_columns_df(categorical_features['monthly_weekday_counts'], 'weekday', 'count')
    
    biba_games_df = pd.concat([biba_games_df, monthly_survey_df, monthly_weekday_counts_df], axis = 1)
    
    #dropping categorical features
    biba_games_df = biba_games_df.drop(columns = list(categorical_features.columns))
    
    #dropping historic hours with low fill rate
    numerical_cols_to_remove = ['historic_hour_0', 'historic_hour_23', 'historic_hour_22', 'historic_hour_21',
                                'historic_hour_7','historic_hour_6','historic_hour_5','historic_hour_4', 
                                'historic_hour_3','historic_hour_2','historic_hour_1', 'MonthYear',
                                'monthly_repeated_sessions', 'historic_repeat_sessions']
    
    biba_games_df = biba_games_df.drop(columns = numerical_cols_to_remove)
    
    impute_biba_games_df = biba_games_df.fillna(0)
    
    #removing the previous columns in the input data
    cols_to_drop = list(full_data.loc[:, 'monthly_number_of_sessions': 'distance_to_nearest_bus_stop'].columns) +\
                   list(full_data.loc[:, 'days_since_first_sess' : 'historic_snow'].columns)
    
    full_data = full_data.drop(columns = cols_to_drop)
    
    #adding processed columns
    full_data = pd.concat([full_data, impute_biba_games_df], axis = 1)
    
    #checking that `historic_hour_*` with low fill rate has been removed
    assert 'historic_hour_0' not in full_data.columns.to_list()
    
    return full_data
    
def preprocess_neighbour(input_data):
    """
    Given the original dataframe, preprocess the columns
    related to locale information (`city` to
    `houses_per_sq_km`). Drop columns with >30%
    NaN values and replace remaining NaN values with 0.
    
    Parameters
    ----------
    input_data : pandas.core.frame.DataFrame
    
    Returns
    -------
    output_data : pandas.core.frame.DataFrame
    """
    
    df_neighbour = input_data.loc[:, 'city':'houses_per_sq_km']
    df_neighbour.drop(columns=['climate'])
    missing = df_neighbour.isna()
    
    # Count number of missing values for each column
    num_missing = missing.sum().sort_values(ascending=False)
    
    # Calculate proportion of missing values for each column
    prop_missing = num_missing / input_data.shape[0]
    
    # Create a list of columns with >30% of values missing
    to_drop = prop_missing[prop_missing > 0.3].index.to_list()
    
    # Add `country` to the list since all playgrounds are in the U.S.
    # Add `city` and `county` since lat. and long. should take care of them
    to_drop.append('country')
    to_drop.append('city')
    to_drop.append('county')
    
    # Drop columns with names in list
    output_data = input_data.drop(to_drop, axis=1)
    
    # Fill in remaining NaN values in locale-related columns with 0
    to_impute = prop_missing[(0 < prop_missing) & (prop_missing <= 0.3)].index.to_list()
    to_impute.remove('city')
    output_data[to_impute] = output_data[to_impute].fillna(0)
    output_data['climate'] = input_data['climate']
    
    # Check that the number of rows is unchanged
    assert input_data.shape[0] == output_data.shape[0]
    
    # Check that `city` column is not in `output_data`
    assert 'city' not in output_data.columns.to_list()

    return output_data
    
def preprocess_weather(input_data):
    """
    Given the original dataframe, preprocess the columns
    related to weather information (`Democrats_08_Votes` to
    the end + `climate`). Impute NaN of `Green_2016` by using values found online, or 0, 
    and replace remaining NaN values with 0.
    
    Parameters
    ----------
    input_data : pandas.core.frame.DataFrame
    
    Returns
    -------
    output_data : pandas.core.frame.DataFrame
    
    """
    
    df_weather = input_data.loc[:, 'Democrats_08_Votes':]
    df_weather['state'] = input_data['state']
    df_weather['climate'] = input_data['climate']
    df_weather['external_id'] = input_data['external_id']
    df_weather['month'] = input_data['month']
    df_weather['year'] = input_data['year']

    
    #fill up NaNs for the `Green_2016` column
    #I only found values for Alaska and North Carolina, so I just put 0 for the other states
    df_weather['Green_2016'] = np.where(
     df_weather['state'] == 'Alaska', 5735, 
         np.where(
            df_weather['state'] == 'North Carolina', 12105,  
             np.where(
                df_weather['Green_2016'].isnull(), 0, df_weather['Green_2016'] 
             )
         )
    )
        
    #Substitute every remaining NaNs by 0
    df_weather = df_weather.fillna(value=0)
    
    output_data = input_data.copy()
    output_data.loc[:, 'Democrats_08_Votes':] = df_weather.loc[:, 'Democrats_08_Votes':]
    output_data['climate'] = df_weather['climate']
    
    #Tests
    
    #Check that there are no missing values in the `Number_of_holidays` column
    if not output_data['Number_of_holidays'].isnull().sum() == 0:
        raise Error('There should not be NaNs in the Number_of_holidays column')

    return output_data
    
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

    #Check that `income_class` column is not in `output_data`
    assert 'income_class' not in X_train_output.columns.to_list()
    assert 'income_class' not in X_valid_output.columns.to_list()

    return (X_train_output, X_valid_output)

def preprocessing_na(input_data):
    """
    Gather all the preprocessing function from the 3 different parts of the data
    
    
    Parameters
    ----------
    input_data : pandas.core.frame.DataFrame
    
    Returns
    -------
    output_data : pandas.core.frame.DataFrame
    
    """

    data_1 = biba_pp(input_data)
    data_2 = preprocess_neighbour(data_1)
    output_data = preprocess_weather(data_2)
    
    #Check that the number of rows is unchanged
    assert input_data.shape[0] == output_data.shape[0]
    
    return output_data

