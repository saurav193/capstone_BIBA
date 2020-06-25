# Data folder

In this folder, you will find `.csv` files and `.zip` files containing the different dataframes we used.

Details : 
- the `dummy` folder contains all the dummy datasets we used in our different tests
- the `columns_names` file contains an empty dataframe with the same columns as the dataset we worked with (in case the main dataset is changed)
- the files containing the word `old` refer to the first dataset we worked with (which wasn't capped to 4000)
- the files containing the word `playground_stats` are the main datasets we worked with
- the files containing `test_data` are our test sets
- the files containing `train_data` are our train/validation sets
- the files containing `processed` correspond to data that have been preprocessed by the script `src/02_preprocessing.py` 
- `X_pred.zip` corresponds to the file that is used as an input of the predicting pipeline
- `processed_pred.zip` is the preprocessed version of `X_pred.zip`
- the files containing `lmer` correspond to the files used in the `src/training_mixed_effects_R.ipynb` notebook