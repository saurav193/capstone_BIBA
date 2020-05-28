import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Build a pipeline for preprocessing data
# Code attribution: DSCI 571 Lecture 8 

# Identify columns to impute with 0
zero_imp_features = []

# Identify columns to impute with mean
mean_imp_features = ['walk_score', 'bike_score', 'Poor_physical_health_days', 'Poor_mental_health_days', 'Adult_smoking']

# Create transformer for 0 imputation
zero_transformer = SimpleImputer(strategy='constant', fill_value=0)

# Create transformer for mean imputation
mean_transformer = SimpleImputer(strategy='mean')

#
# Create transformer for `Republicans_08_Votes`
rep_08_votes_transformer = SimpleImputer(strategy='constant', fill_value=193841)

# Create transformer for `Democrats_08_Votes`
dem_08_votes_transformer = SimpleImputer(strategy='constant', fill_value=123594)

# Create transformer for `Republican_12_Votes`
rep_12_votes_transformer = SimpleImputer(strategy='constant', fill_value=164676)

# Create transformer for `Democrats_12_Votes`
dem_12_votes_transformer = SimpleImputer(strategy='constant', fill_value=122640)

# Create transformer for `Republicans_2016`
rep_2016_transformer = SimpleImputer(strategy='constant', fill_value=163387)

# Create transformer for `Democrats_2016`
dem_2016_transformer = SimpleImputer(strategy='constant', fill_value=116454)

# Create transformer for `Libertarians_2016`
lib_2016_transformer = SimpleImputer(strategy='constant', fill_value=18725)

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('zero', zero_transformer, zero_imp_features),
        ('mean', mean_transformer, mean_imp_features),
        ('rep_08_votes', rep_08_votes_transformer, 'Republicans_08_Votes'),
        ('dem_08_votes', dem_08_votes_transformer, 'Democrats_08_Votes'),
        ('rep_12_votes', rep_12_votes_transformer, 'Republican_12_Votes'),
        ('dem_12_votes', dem_12_votes_transformer, 'Democrats_12_Votes'),
        ('rep_2016', rep_2016_transformer, 'Republicans_2016'),
        ('dem_2016', dem_2016_transformer, 'Democrats_2016'),
        ('lib_2016', lib_2016_transformer, 'Libertarians_2016')
    ]
)