
#Download libraries
import pandas as pd
import numpy as np

# Author: Tani
def add_lagged_target(df):
    """
    This function takes a dataframe with the columns 
    "external_id", "month", "year" as primary keys.
    Adds a column of the "unacast_seassion_count" at lag of 1, as "session_lagged_1",
    and then deletes the first occurrence for each playground.


    Parameters
    ----------------
    df : pd.DataFrame
       A dataframe containing the columns "external_id", "month", "year"
       and "unacast_seassion_count"


    Returns
    ----------------
    pd.DataFrame
        With the new lagged session column, and deleted first occurrence of each playground, sorted by ["external_id","year","month"]
    """
    # subset and sort
    lagged = df.loc[:,["external_id","month","year","unacast_session_count"]].sort_values(by=["external_id","year","month"])

    # creat new column shifted by one (after sorting)
    lagged["session_lagged_1"] = lagged['unacast_session_count'].shift(1)
    
    # join the new column into the general dataframe
    out = pd.merge(df,lagged,how='left', left_on=['external_id','month','year',"unacast_session_count"], right_on =['external_id','month','year',"unacast_session_count"]).sort_values(by=["external_id","year","month"])

    # identify the 1st row of each playground, and subset as a df
    to_del = out.sort_values(by=["external_id","year","month"]).groupby('external_id',as_index=False).nth(0)
    # join the df with the rows to delete, then delete all rows that are have duplicates (both copies)
    out = pd.concat([out,to_del]).drop_duplicates(keep=False)
    return out

# Need to add unit tests
# test for correct number of rows deleted
# test for only one added column
# test randomly for correct values?
