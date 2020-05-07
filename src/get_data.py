# author: Sirine	Chahma
# date: 2020-05-07

'''This script downloads the data from a given url and saves in the data
folder in the project directory. This script takes a url to the data and a 
file location as the arguments.
Usage: src/get_data.py --url=<url> --file_location=<file_location>
 
'''

import requests
from docopt import docopt
import pandas as pd
import numpy as np

opt = docopt(__doc__)

def main(url, file_location):
    # download and save data
    r = requests.get(url)
    with open(file_location, "wb") as f:
        f.write(r.content) 
    
    test_url(url)
    #test(file_location)

def test(file_location):
    df = pd.read_csv(file_location,  sep=";")
    assert df.shape[0] != 0
    print(f"file successfully saved to {file_location}")
    
def test_url(url):
   test = np.DataSource()
   assert test.exists(url), "Invalid URL!"


if __name__ == "__main__":
    main(opt["--url"], opt["--file_location"])
