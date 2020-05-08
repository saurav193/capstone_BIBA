# author: Sirine Chahma
# date: 2020-05-07
#
"""This script splits the data given as an input into training and test data.
It saves the two sets in two separate csv files, in the file that is given as an input.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 
Usage: split_data.py
"""

#import dependencies
import pandas as pd

def main():
    #Read the data and split between train and test set
    data = pd.read_csv('data/playground_stats.zip')
    test_data = data[(data['year']==2019) & (data['month'] > 9)]
    train_data = data[~data.index.isin(test_data.index)]
    
    test_split(data, train_data, test_data)

    #Save the files
    compression_opts = dict(method='zip',archive_name='out.csv')  
    test_data.to_csv("data/test_data.zip", index = False, compression=compression_opts)
    train_data.to_csv("data/train_data.zip", index = False, compression=compression_opts)
    print('Split successful!')

    
def test_split(data, train_data, test_data):
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0], "The split is wrong"
    

if __name__ == "__main__":
    main()
