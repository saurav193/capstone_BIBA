# Date: 2020-05-29
#
# The function that can be found on this script can be used as part of the data preprocessing process.
# The function can be used to drop columns that we think are irrelevant for our analysis.

import pandas as pd
import re

def drop_columns(input_data):
    """
    Drops columns that are deemed irrelevant from the original dataframe.
    This list of columns to drop was derived from EDA and 
    consultation with the Biba team.
    
    Parameters
    ---------------
    input_data: pandas.core.frame.DataFrame
        Data to which imputation and feature engineering have already been applied.

    Returns
    ---------------
    pandas.core.frame.DataFrame
        
    """
    data = input_data.copy()

    data = data.drop(columns = ['external_id', 'monthly_count_of_holidays', 'B13016e2', 'B19113e1', 'name',
                                'city', 'state', 'country', 'county', 'MonthYear', 'date', 
                                'streets_per_node_counts_0', 'streets_per_node_counts_0_osid', 
                                'streets_per_node_counts_0_osdw', 'self_loop_proportion', 'self_loop_proportion_osid', 
                                'self_loop_proportion_osdw', 'circuity_avg', 'circuity_avg_osid', 'circuity_avg_osdw', 
                                'clean_intersection_density_km', 'node_density_km', 'clean_intersection_count_osid', 
                                'node_density_km_osdw', 'intersection_density_km_osdw', 'street_density_km_osid', 
                                'edge_density_km_osid', 'intersection_density_km_osid', 'node_density_km_osid', 
                                'edge_density_km_osdw', 'street_density_km_osdw', 'clean_intersection_count', 
                                'clean_intersection_count_osdw', 'clean_intersection_density_km_osdw', 'street_density_km', 
                                'edge_density_km', 'intersection_density_km', 'clean_intersection_density_km_osid', 
                                'streets_per_node_counts_8', 'streets_per_node_proportion_8', 'streets_per_node_proportion_7_osid', 
                                'streets_per_node_counts_7_osid', 'streets_per_node_proportion_8_osdw', 
                                'streets_per_node_counts_8_osdw', 'streets_per_node_proportion_7', 'streets_per_node_counts_7', 
                                'streets_per_node_counts_7_osdw', 'streets_per_node_proportion_7_osdw', 
                                'streets_per_node_proportion_6_osid', 'streets_per_node_counts_6_osid', 
                                'streets_per_node_proportion_6', 'streets_per_node_counts_6', 'streets_per_node_counts_6_osdw', 
                                'streets_per_node_proportion_6_osdw', 'transit_score', 'closest_place_category', 
                                'closest_place_distance'])
    
    # Gather all `temp_min_*` columns
    temp_list = [i for i in data.columns if re.match('temp_min_*', i)]
    
    # Gather all `streets_per_node_proportion_*` columns
    street_list = [i for i in data.columns if re.match('streets_per_node_proportion_*', i)]

    # Gather all news events columns
    news_state_list = data.loc[:, 'total_events_across_state':'material_conflict_events_across_state'].columns.to_list()
    news_radius_list = data.loc[:, 'total_events_500_meters':'material_conflict_events_2000_meters'].columns.to_list()
    
    # Gather other irrelevant census columns
    sex_age_list = ['B01001e27', 'B01001e28', 'B01001e29', 'B01001e3', 'B01001e30', 'B01001e4', 'B01001e5', 'B01001e6']
    
    # Gather all Biba survey columns
    monthly_survey_list = data.loc[:, ['monthly_weekday_counts', 'monthly_survey']].columns.to_list()
    historic_survey_list = data.loc[:, 'historic_weekday_0':'historic_variety'].columns.to_list()
    
    cols_to_drop =  temp_list + street_list + news_state_list + news_radius_list + sex_age_list \
                    + monthly_survey_list + historic_survey_list

    data = data.drop(columns = cols_to_drop)
    
    test_drop(input_data, data)
    
    return data

def test_drop(input_data, data):
    """
    Test for the drop_columns() function
    
    Parameters
    ---------------
    input_data: pandas.core.frame.DataFrame
        The input dataframe of the drop_columns() function
    data: pandas.core.frame.DataFrame
        The output dataframe of the drop_columns() function
    
    Returns
    ---------------
    Error if the drop_columns() function doesn't work the expected way
        
    """
    # check that the number of rows is unchanged
    assert input_data.shape[0] == data.shape[0]

    # check that 'temp_min_*' columns have been removed from the output
    assert 'temp_min_35_45' not in list(data.columns)

    # check that there are fewer columns in the output
    assert input_data.shape[1] > data.shape[1]

def drop_missing_unacast(raw_data):
    """
    Given the raw input data, return a dataframe where
    rows missing the target variable have been removed.

    Parameters
    ----------
    raw_data: pandas.core.frame.DataFrame
        Raw input data

    Returns
    -------
    pandas.core.frame.DataFrame

    """
    output_data = raw_data.dropna(axis=0, subset=['unacast_session_count'])

    # check that there are no NaN values in the target
    assert output_data['unacast_session_count'].isna().sum() == 0

    # check that the number of columns is unchanged
    assert raw_data.shape[1] == output_data.shape[1]

    return output_data
