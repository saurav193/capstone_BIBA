
# importing libraries
import pandas as pd
import numpy as np
import re

def comb_cols(input_df):

    """
    This function takes the entire data(all cols) and combines some of them
    into new features using expert knowledge.

    Parameters
    ---------------
    input_df : pandas.DataFrame
       the entire dataframe with all the columns that has been imputed
        
    Returns
    ---------------
    pandas.DataFrame
        with new features combining some existing features
    """
    output_df = input_df.copy()
    
    # combining the count of equipments
    
    equipments = ['slide', 'climb', 'tube', 'overhang', 'bridge', 'swing', 'obsta', 'crawls']
    
    #creating list of columns to combine. 2(monthly and historic) for each type of equipment
    new_cols_list = {}
    cols_to_drop = []
    
    for equipment in equipments:
        new_cols_list["monthly_"+equipment] = [i for i in input_df.columns if re.match('monthly_.*'+equipment+'.*', i)]
        new_cols_list["historic_"+equipment] = [i for i in input_df.columns if re.match('historic_.*'+equipment+'.*', i)]
        if new_cols_list["monthly_"+equipment]==[]:
            new_cols_list.pop("monthly_"+equipment)
            
    for key, val in new_cols_list.items():
        output_df[key+"_count_comb"] = np.sum(output_df.loc[:, val], axis = 1)
        cols_to_drop = cols_to_drop + val # dropping the columns combined together
        
    
    # combining wind speed cols
    
    output_df['avg_wind_calm'] = input_df['avg_wind_0_1']
    output_df['avg_wind_light_air'] = np.sum(input_df.loc[:, ['avg_wind_1_2','avg_wind_2_3','avg_wind_3_4']], axis = 1)
    output_df['avg_wind_light_br'] = np.sum(input_df.loc[:, ['avg_wind_4_5','avg_wind_5_6','avg_wind_6_7','avg_wind_7_8']], axis = 1)
    output_df['avg_wind_gentle_br'] = np.sum(input_df.loc[:, ['avg_wind_8_9','avg_wind_9_10','avg_wind_10_11','avg_wind_11_12']], axis = 1)
    output_df['avg_wind_moderate_br'] = input_df['avg_wind_12_above']
    
    output_df['monthly_ws_calm'] = input_df['monthly_ws_below_2']
    output_df['monthly_ws_light_air'] = input_df['monthly_ws_2_to_4']
    output_df['monthly_ws_light_br'] = np.sum(input_df.loc[:, ['monthly_ws_4_to_6','monthly_ws_6_to_8']], axis = 1)
    output_df['monthly_ws_gentle_br'] = np.sum(input_df.loc[:, ['monthly_ws_8_to_10','monthly_ws_10_to_12']], axis = 1)
    output_df['monthly_ws_moderate_br'] = np.sum(input_df.loc[:, ['monthly_ws_12_to_14','monthly_ws_14_to_16','monthly_ws_above_16']], axis = 1)
    output_df['historic_ws_calm'] = input_df['historic_ws_below_2']
    output_df['historic_ws_light_air'] = input_df['historic_ws_2_to_4']
    output_df['historic_ws_light_br'] = np.sum(input_df.loc[:, ['historic_ws_4_to_6','historic_ws_6_to_8']], axis = 1)
    output_df['historic_ws_gentle_br'] = np.sum(input_df.loc[:, ['historic_ws_8_to_10','historic_ws_10_to_12']], axis = 1)
    output_df['historic_ws_moderate_br'] = np.sum(input_df.loc[:, ['historic_ws_12_to_14','historic_ws_14_to_16','historic_ws_above_16']], axis = 1)
    
    ##### other columns to be added ####
    
    # averaging fertility
    output_df['avg_fertility_rate'] = np.mean(input_df.loc[:, 'fertility_rate_2003':'fertility_rate_2018'], axis=1)

    
    # dropping already combined cols
    
    output_df = output_df.drop(columns = ['avg_wind_0_1','avg_wind_1_2','avg_wind_2_3','avg_wind_3_4','avg_wind_4_5',
                                            'avg_wind_5_6','avg_wind_6_7','avg_wind_7_8','avg_wind_8_9',
                                            'avg_wind_9_10','avg_wind_10_11','avg_wind_11_12','avg_wind_12_above',
                                            'monthly_ws_below_2','monthly_ws_2_to_4','monthly_ws_4_to_6','monthly_ws_6_to_8',
                                            'monthly_ws_8_to_10','monthly_ws_10_to_12','monthly_ws_12_to_14','monthly_ws_14_to_16',
                                            'monthly_ws_above_16','historic_ws_below_2','historic_ws_2_to_4','historic_ws_4_to_6',
                                            'historic_ws_6_to_8','historic_ws_8_to_10','historic_ws_10_to_12','historic_ws_12_to_14',
                                            'historic_ws_14_to_16','historic_ws_above_16']
    output_df = output_df.drop(columns = ['fertility_rate_2003','fertility_rate_2004','fertility_rate_2005','fertility_rate_2006',
                                          'fertility_rate_2007','fertility_rate_2008','fertility_rate_2009','fertility_rate_2010',
                                          'fertility_rate_2011','fertility_rate_2012','fertility_rate_2013','fertility_rate_2014',
                                          'fertility_rate_2015','fertility_rate_2016','fertility_rate_2017','fertility_rate_2018',]
   
    output_df = output_df.drop(columns = cols_to_drop)                        
    
    
    return output_df
    