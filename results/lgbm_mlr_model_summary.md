## Report on Linear Regression, Light GBM and Mixed Linear Effects model

### 1. Initial round with 0 imputation

**Results**

|Model| Test RMSE | Comments|
|-----|-----------|---------|
| Simple LR |  668.082 | Most basic model |
| Ridge with gridsearch| 667.956 | Same performance as simple LR |
| Simple SVR| NA | Very long runtime |
| Simple LGBM | 590.936 | Poor performance but better than LR|
| Simple LGBM dropping repeated counts(monthly and historic)| 306.431 | Big improvement observed |
| LGBM with MAE objective funct | 303.248 | Performs better |
| LGBM with MAE objective funct and scaled X | 301.389 | Scaling of X does not improve the model
| Ridge with log transformed y | 1.011 | Performs better, but cannot compare with Non-log transformed | 
| LGBM with log transformed y | 0.711 | Same as above |
| LGBM with PCA columns | 5689.688 | Very high bias and poor result in validation set |

**Verdict**

- LGBM seems to perfomr much better than simple regression
- Dropping of correlated columns in LGBM might imrpove its performance
- Hyperparameter optimization also might improve performance
- PCA is NOT suitable for this dataset.


### 2. Second round with extensive feature engineering and proper imputations and capping target variable

**Results** 

|Model| Test MAE| Test MSE|
|-----|---------|---------|
| Simple LR | 99.675 | 190.922| 
| Ridge regression| 99.631 |190.895|
| Simple LGBM | 53.161 | 110.994 |
| LGBM with gridsearch | NA | 104.5832174 |
| LGBM with log transformed y | 51.2 | NA |
| LGBM with best hyperparameters | 40.871 | 96.991 |
| LGBM with best hyperparameters and dropped repeated counts| 40.377 | 95.334 |
| LMER model with State as random effect | 107.94 | 199.76 |
| LMER model with Density Class as random effect | 108.63 | 199.70 |
| LMER model with Income Class as random effect | 108.69 | 199.75 |
| LMER model with Climate as random effect | 109.57 | 200.43 |
| LMER model with Kmeans-2-clusters as random effect | 108.92 | 199.81 |
| LMER model with Kmeans-4-clusters as random effect | 109.02 | 199.73 |

**Verdict**

- LGBM model might be improved by removing some features using RFE
- LMER is not suitable as it performs same as linear regression

### 3. Possible next steps
- Run RFE to select features for LGBM
- Run LMER with Kmeans clustering







