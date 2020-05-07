CAPSTONE Machine Learning
====

# Branching Strategy

A branch has been created for each student. These branches should be treated as
the 'master' branch for that student.

# Usage

To download the data from a URL 

```
python src/get_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv" --file_location=data/raw_data.csv
```

To split the data between train and test set

```
python src/split_data.py --input_file=data/raw_data.csv --output_dir=data
```


