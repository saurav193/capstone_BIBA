# author: Sirine Chahma
# date: 2020-05-07
#
'''  This script splits the data given as an input into training and test data.
It saves the two sets in two seperate csv files, in the file that is given as an input.
This script takes two arguments : a path/filename pointing to the data to be read in
and a path/filename pointing to where the cleaned data should live. 
Usage: split_data.py --input_file=<input_file> --output_dir=<output_dir>
Options:
--input_file=<input_file>    Path (including filename) to raw data (csv file)
--output_dir=<output_dir>   Path to directory where the processed data should be written
''' 

#import dependencies
from docopt import docopt
import pandas as pd

opt = docopt(__doc__)


def main(input_file, output_dir):
    data = pd.read_csv(input_file)
    test_data = data[(data['year']==2019) & (data['month'] > 9)]
    train_data = data[~data.index.isin(test_data.index)]
    
    test_split()
    
    test_data.to_csv(output_dir+"/" + "test_data.csv", index = False, compression='gzip')
    train_data.to_csv(output_dir+"/" + "train_data.csv", index = False, compression='gzip')
    print('Split successful!')

    
def test_split():
    assert train_data.shape[0] + test_data.shape[0] == data.shape[0], "The split is wrong"
    

if __name__ == "__main__":
    main(opt["--input_file"], opt["--output_dir"])
