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
    to_impute.remove('county')
    output_data[to_impute] = output_data[to_impute].fillna(0)
    output_data['climate'] = input_data['climate']

    return output_data
    
def preprocess_weather(input_data):
    """
    Given the original dataframe, preprocess the columns
    related to weather information (`Democrats_08_Votes` to
    the end + `climate`). Impute NaN of `Number_of_holidays` 
    by using the values the we have for the same month,
    impute NaN of `Green_2016` by using values found online, or 0, 
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
    
    
    #fill up NaNs for `Number_of_holidays` column
    #I sorted the values so that the values are ordered by time, and the NaNs are at the end of each time period
    df_weather = df_weather.sort_values(['month', 'year', 'Number_of_holidays'])
    df_weather['Number_of_holidays'] = df_weather['Number_of_holidays'].fillna(method='ffill')
    
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
    
    df_weather['climate'] = df_weather['climate'].fillna(df_weather['climate'].mode()[0])
    
    #Substitute every remaining NaNs by 0
    df_weather = df_weather.fillna(value=0)
    
    output_data = input_data.copy()
    output_data.loc[:, 'Democrats_08_Votes':] = df_weather.loc[:, 'Democrats_08_Votes':]
    output_data['climate'] = df_weather['climate']
    
    #Tests
    
    #Check that there are no missing values in the `Number_of_holidays` column
    if not output_data['Number_of_holidays'].isnull().sum() == 0:
        raise Error('There should not be NaNs in the Number_of_holidays column')
    
    #Check that every month has only one value for the `Number_of_holiday` column
    number_of_error = 0
    for month in range(12):
        for year in [2018, 2019]:
            sub_df = output_data[(output_data['month'] == month+1) & (output_data['year'] == year)]
            if len(sub_df['Number_of_holidays'].unique()) > 1:
                number_of_error += 1 
    if not number_of_error == 0:
        raise Error('Every month should have the same value for Number_of_holidays')
    
    
               
    return output_data
    
def clean_categorical(input_data, to_drop=['income_class', 'density_class', 'climate']):
    """
    Given the original dataframe, uses One-Hot-Encoding to encode the categorical variables
    
    
    Parameters
    ----------
    input_data : pandas.core.frame.DataFrame
    to_drop : list
        The list of the categorical variables on which we want to apply OHE
    
    Returns
    -------
    output_data : pandas.core.frame.DataFrame
    
    """
    
    output_data = input_data.copy()

    #Apply One-Hot-Encoding to each one of the categorical variable
    for col in to_drop:
        ohe = OneHotEncoder(sparse=False, dtype=int)
        sub_df = pd.DataFrame(ohe.fit_transform(input_data[[col]]), columns=ohe.categories_[0])
        output_data = pd.concat((output_data, sub_df), axis=1)
    #Drop the columns for which we used OHE
    output_data.drop(columns = to_drop, inplace=True)
    
    return output_data

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
    return output_data

