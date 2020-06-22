# Using Machine Learning to Predict Playground Usage Across the U.S.

A capstone project for the UBC Master of Data Science program

Authors: Saurav Chowdhury, Sirine Chahma, Reiko Okamoto, Tani Barasch

## About

## Report
- Link to proposal
- Link to final report

## Usage
- How to manage pipenv virtual environment
- How to use Makefile 1 (analysis pipeline)
- How to render the RMarkdown file
- How to use Makefile 2 (prediction pipeline)
- Run each script independently
### Split the data
`python src/01_split_data.py --in_path=data/playground_stats.csv --out_path=data`

### Preprocessing
`python src/02_preprocessing.py --test=data/test_data.zip --train=data/train_data.zip`

### GBR
`python src/03_gbr_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib/ --out_path=results/`

### Catboost
`python src/04_catboost_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib --out_path=results/`

### LGBM
`python src/05_lgbm_model.py --train=data/processed_train.zip --test=data/processed_test.zip --model_path=src/joblib/ --out_path=results/`

## Dependencies
- Python 3.7 and packages
- R and packages

## References
