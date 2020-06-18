# author: Sirine Chahma
# date: 2020-05-07
#
"""This script splits the data given as an input into training and test data.
It saves the training and test sets in two separate zip files, in the folder that is given as an input.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path pointing to where the cleaned data should live. 

Usage: src/split_data.py --in_path=<in_path> --out_path=<out_path>

Options:
--in_path=<in_path>             The path (including filename) of the original dataset file in zip/csv format, relative to the root 
--out_path=<out_path>               The path (without a '/' at the end) of the folder where you want to set the train/test sets, relative to the root
"""

#import dependencies
import pandas as pd
from docopt import docopt
import os

opt = docopt(__doc__)

def main(in_path, out_path):
    #Read the data and split between train and test set
    data = pd.read_csv(in_path)
    test_data = data[(data['year']==2019) & (data['month'] > 9)]
    train_data = data[~data.index.isin(test_data.index)]
    #Comment the following line to keep the January 2018 data
    train_data = train_data.query("month != 1 | year != 2018")
    # Comment the following line if the playground 'CA00070678' has been removed
    train_data = train_data.query("external_id != 'CA00070678'")
    test_data = test_data.query("external_id != 'CA00070678'")

    
    test_split(data, train_data, test_data, out_path)

    #Save the files
    compression_opts = dict(method='zip',archive_name='out.csv')  
    test_data.to_csv(out_path + "/test_data.zip", index = False, compression=compression_opts)
    train_data.to_csv(out_path + "/train_data.zip", index = False, compression=compression_opts)
    data.to_csv(out_path + "/playground_stats.zip", index = False, compression=compression_opts)

    test_split(data, train_data, test_data, out_path)

    print('Split successful!')

    
def test_split(data, train_data, test_data, out_path):
    """
    This functions checks if the main function is able to split the data into train set and test set
    """
    #Check that the dimmensions match
    #Delete the `+ data.query("month == 1 & year == 2018").shape[0]` for the sum if you want to keep the January 2018 data 
    assert train_data.shape[0] + test_data.shape[0] + data.query("month == 1 & year == 2018").shape[0] + data.query("external_id == 'CA00070678'").shape[0] - 1 == data.shape[0], "The split is wrong"
    assert train_data['external_id'].unique().shape[0] == test_data['external_id'].unique().shape[0], "The numbers of unique playground in the train set and the test set are different"
    assert data['external_id'].unique().shape[0] == train_data['external_id'].unique().shape[0] + 1, "The numbers of unique playground in the train set and the original set are different"


if __name__ == "__main__":
    main(opt["--in_path"], opt["--out_path"])
