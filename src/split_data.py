# author: Sirine Chahma
# date: 2020-05-07
#
"""This script splits the data given as an input into training and test data.
It saves the two sets in two separate csv files, in the file that is given as an input.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 
Usage: src/split_data.py --input_file=<input_file> --output_folder=<output_folder>
"""

#import dependencies
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(input_file, output_folder):
    #Read the data and split between train and test set
    data = pd.read_csv(input_file)
    test_data = data[(data['year']==2019) & (data['month'] > 9)]
    train_data = data[~data.index.isin(test_data.index)]
    #Comment the following line to keep the January 2018 data
    train_data = train_data.query("month != 1 | year != 2018")
    # Comment the following line if the playground 'CA00070678' has been removed
    train_data = train_data.query("external_id != 'CA00070678'")
    test_data = test_data.query("external_id != 'CA00070678'")

    
    test_split(data, train_data, test_data)

    #Save the files
    compression_opts = dict(method='zip',archive_name='out.csv')  
    test_data.to_csv(output_folder + "/test_data.zip", index = False, compression=compression_opts)
    train_data.to_csv(output_folder + "/train_data.zip", index = False, compression=compression_opts)
    data.to_csv(output_folder + "/playground_stats.zip", index = False, compression=compression_opts)

    print('Split successful!')

    
def test_split(data, train_data, test_data):
    #Delete the `+ data.query("month == 1 & year == 2018").shape[0]` for the sum if you want to keep the January 2018 data 
    assert train_data.shape[0] + test_data.shape[0] + data.query("month == 1 & year == 2018").shape[0] + data.query("external_id == 'CA00070678'").shape[0] - 1 == data.shape[0], "The split is wrong"
    assert train_data['external_id'].unique().shape[0] == test_data['external_id'].unique().shape[0], "The numbers of unique playground in the train set and the test set are different"
    assert data['external_id'].unique().shape[0] == train_data['external_id'].unique().shape[0] + 1, "The numbers of unique playground in the train set and the original set are different"

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_folder"])
