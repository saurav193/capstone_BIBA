# date: 2020-05-29
#
# The function that can be found on this script can be used as part of the data preprocessing process.
# The function can be used to dop columns that we think are irrelevant for our analysis.

#import dependencies
import pandas as pd
import re

def drop_columns(input_data):
    """
    Drops some columns that we think are irrelevant from the original dataframe.
    
    Parameters
    ---------------
    
    input_data : pandas.core.frame.DataFrame
    
    Returns
    ---------------
    pandas.core.frame.DataFrame
        
    """
    data = input_data.copy()
    data = data.drop(columns = ['external_id', 'monthly_count_of_holidays', 'B13016e2', 'B19113e1', 'name', 
                            'MonthYear', 'date', 'streets_per_node_counts_0', 'streets_per_node_counts_0_osid', 
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
                            'closest_place_distance', 'monthly_weekday_counts', 'monthly_survey', 'historic_weekday_0',
                            'historic_weekday_1', 'historic_weekday_2', 'historic_weekday_3', 'historic_weekday_4', 
                            'historic_weekday_5', 'historic_weekday_6', 'historic_accessible', 'historic_allages',
                            'historic_cleanliness', 'historic_condition', 'historic_regular', 'historic_revisit', 
                            'historic_safety', 'historic_travel', 'historic_variety', 'city', 'state', 'county', 'country'])
    temp_list = [i for i in data.columns if re.match('temp_min_*', i)]
    street_list = [i for i in data.columns if re.match('streets_per_node_proportion_*', i)]
    data = data.drop(columns=temp_list)
    data = data.drop(columns=street_list)
    cols_to_drop = list(data.loc[:, 'avg_impact_of_events_2000_meters':'material_conflict_events_2000_meters'].columns) +\
                       list(data.loc[:, 'B01001e27': 'B01001e6'].columns)

    data = data.drop(columns = cols_to_drop)
    test_drop(input_data, data)
    return data

def test_drop(input_data, data):
    """
    Test for the drop_columns() function 
    
    Parameters
    ---------------
    
    input_data : pandas.core.frame.DataFrame
        The input dataframe of the drop_columns() function
    data : pandas.core.frame.DataFrame
        The output dataframe of the drop_columns() function
    
    
    Returns
    ---------------
    Error if the drop_columns() function doesn't work the expected way
        
    """
    assert input_data.shape[0] == data.shape[0]
    assert 'temp_min_35_45' not in list(data.columns)
    assert input_data.shape[1] > data.shape[1]

