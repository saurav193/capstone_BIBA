# author: Tani Barasch
# date: 2020-05-20
#



#import dependencies
import pandas as pd
import numpy as np
from collections import defaultdict
import re

# function for parsing dict columns into the data frame
def dict_to_columns_df(col, key, val):
    """
    This functions takes a dataframe column which is in the
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

# function for dealing with weather data, columns ###-###
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
    
    #drop the rows with NaNs in the 'unacast_session_count` column - shifted to main
    #input_data = input_data.dropna(subset=['unacast_session_count'])
    
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
    
    #Substitute every remaining NaNs by 0
    df_weather = df_weather.fillna(value=0)
    
    output_data = input_data.copy()
    output_data.loc[:, 'Democrats_08_Votes':] = df_weather.loc[:, 'Democrats_08_Votes':]
    output_data['climate'] = df_weather['climate']
    
    return output_data

#func to deal with locale information (`city` to `houses_per_sq_km`)
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
    missing = df_neighbour.isna()
    
    # Count number of missing values for each column
    num_missing = missing.sum().sort_values(ascending=False)
    
    # Calculate proportion of missing values for each column
    prop_missing = num_missing / df.shape[0]
    
    # Create a list of columns with >30% of values missing
    to_drop = prop_missing[prop_missing > 0.3].index.to_list()
    
    # Add `country` to the list since all playgrounds are in the U.S.
    to_drop.append('country')
    
    # Drop columns with names in list
    output_data = input_data.drop(to_drop, axis=1)
    
    # Fill in remaining NaN values with 0
    output_data = output_data.fillna(0)
    
    return output_data

def main(df):
    

    


    # drop the rows with NaNs in the 'unacast_session_count` column
     df = df.dropna(subset=['unacast_session_count'])
    # weather columns
    weather_col = df.loc[:,'Democrats_08_Votes':].columns.tolist()




    #Save the files
    compression_opts = dict(method='zip',archive_name='out.csv')  
    df.to_csv("data/pre_process_v1.zip", index = False, compression=compression_opts)
    print('Pre-processing successful!')

def test_preprocess(df):



if __name__ == "__main__":
    main()