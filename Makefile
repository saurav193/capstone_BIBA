# author: Saurav Chowdhury, Sirine Chahma, Reiko Okamoto, Tani Barasch
# date: 2020-06-18

report : results/lgbm_train_result.csv results/catboost_train_result.csv results/gbr_train_result.csv
predict : results/predicted_data.csv


# Split the data
# 
#This script splits the data into train and test sets. The test set corresponds to the data collected between October and December 2019
data/test_data.zip data/train_data.zip data/playground_stats.zip : src/01_split_data.py data/playground_stats.csv
	python src/01_split_data.py --in_path=data/playground_stats.csv --out_path=data
	
# Preprocess the data
# 
#This script preprocesses and saves the train and test sets as zip files. It can also preprocess new data using the imputer and one-hot encoder that were previously fitted to the training data. It also saves dummy train and test sets as zip files that can be used to test functions.
src/joblib/imputer.joblib src/joblib/ohe.joblib data/processed_train.zip data/processed_test.zip data/dummy/dummy_test_data.zip data/dummy/dummy_train_data.zip : src/02_preprocessing.py data/test_data.zip data/train_data.zip 
	python src/02_preprocessing.py --test=data/test_data.zip --train=data/train_data.zip

# GradientBoostingRegressor
# 
# This script fits a model using sklearn's GradientBoostingRegressor, saves the fitted model, and outputs the MAE values as a CSV file.
results/gbr_train_result.csv src/joblib/gbr_model.joblib : src/03_gbr_model.py data/processed_train.zip data/processed_test.zip data/dummy/dummy_test_data.zip data/dummy/dummy_train_data.zip
	python src/03_gbr_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib/ --out_path=results/

# Catboost
# 
# This script fits a model using CatBoost, saves the fitted model, and outputs the MAE values as a CSV file.
results/catboost_train_result.csv src/joblib/catboost_model.joblib : src/04_catboost_model.py data/processed_train.zip data/processed_test.zip data/dummy/dummy_test_data.zip data/dummy/dummy_train_data.zip
	python src/04_catboost_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib/ --out_path=results/

# LightGBM
# 
# This script fits a model using LightGBM, saves the fitted model, and outputs the MAE values as a CSV file.
results/lgbm_train_result.csv src/joblib/lgbm_model.joblib : src/05_lgbm_model.py data/processed_train.zip data/processed_test.zip data/dummy/dummy_test_data.zip data/dummy/dummy_train_data.zip
	python src/05_lgbm_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib/ --out_path=results/


##############################
# Prediction Pipeline
##############################

# Preprocessing of the new data

data/processed_pred.zip data/dummy/dummy_pred_data.zip : src/02_preprocessing.py src/joblib/imputer.joblib src/joblib/ohe.joblib data/X_pred.zip
	python src/02_preprocessing.py --test=data/X_pred.zip

# Predictions
# 
# This scripts predicts the unacast_session_count for the file named 'X_pred'
results/predicted_data.csv : src/07_prediction.py src/joblib/lgbm_model.joblib src/joblib/catboost_model.joblib src/joblib/gbr_model.joblib data/processed_pred.zip
	python src/07_prediction.py --new_data=data/processed_pred.zip


	
# cleaning everything
clean :
	rm -rf results/lgbm_train_result.csv
	rm -rf src/joblib/lgbm_model.joblib
	rm -rf results/catboost_train_result.csv
	rm -rf src/joblib/catboost_model.joblib
	rm -rf results/gbr_train_result.csv
	rm -rf src/joblib/gbr_model.joblib
	rm -rf src/joblib/imputer.joblib
	rm -rf src/joblib/ohe.joblib 
	rm -rf data/processed_train.zip 
	rm -rf data/processed_test.zip 
	rm -rf data/dummy/dummy_test_data.zip 
	rm -rf data/dummy/dummy_train_data.zip
	rm -rf data/test_data.zip
	rm -rf data/train_data.zip
	rm -rf data/playground_stats.zip
	rm -rf data/dummy/dummy_pred_data.zip
