# date: 2020-05-29
#
"""This script drops blabla
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 
Usage: src/drop.py
"""

#import dependencies
import pandas as pd

def drop(data):
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
                            'closest_place_distance'])
    temp_list = [i for i in data.columns if re.match('temp_min_*', i)]
    street_list = [i for i in data.columns if re.match('streets_per_node_proportion_*', i)]
    data = data.drop(columns=temp_list)
    data = data.drop(columns=street_list)
    cols_to_drop = list(data.loc[:, 'avg_impact_of_events_2000_meters':'material_conflict_events_2000_meters'].columns) +\
                       list(data.loc[:, 'B01001e27': 'B01001e6'].columns)

    data = data.drop(columns = cols_to_drop)
    return data

