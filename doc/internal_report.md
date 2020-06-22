Internal report
================
Saurav Chowdhury, Sirine Chahma, Reiko Okamoto, Tani Barasch
17/06/2020

## Purpose

This report serves four purposes: (1) help individuals navigate our
GitHub repository; (2) discuss our analysis and findings; (3) report the
performances of selected models: and (4) provide recommendations on how
to move forward.

## Description of the data

Exploratory data analysis was performed at the beginning of the project.
The code for the exploratory data analysis can be found in the
[`/eda`](https://github.com/Z2hMedia/capstone_machine_learning/tree/master/eda)
directory.

The key observations are as follows. First, the marginal distribution of
the `unacast_session_count` has a positive skew (Figure 1). Second,
Figure 2 shows the sparsity of the data. Many of the features derived
from data collected through the app are sparse. Third, missing values
were present in both the explanatory and response variables. The
presence of missing values across the explanatory variables is
summarized in Figure 3. The next two histograms illustrate the
distribution of missing `unacast_session_count`. As shown in Figure 4,
there are a handful of playgrounds that are missing the target value for
over half of the months. Figure 5 suggests that there is a temporal
pattern in the distribution of missing target values.
`unacast_session_count` is more likely to be missing in the winter
months; notably, the target value for January 2018 is missing for many
playgrounds.

Exploratory data analysis also revealed the possible duplication of
information in the dataset. For example, among the features derived from
the U.S. Census, information related to sex is encoded in several places
(i.e. “Sex by Age”, “Sex by Marital Status”, “Sex by School
Enrollment”). Upon closer inspection of the dataset, we also
discovered that some columns are merely the sum of others. The pattern
can be observed among the U.S. Census-related features; for example,
`B13016e2`(“Women Who Had a Birth by Age: Total”) is the sum of
`B13016e3` through `B13016e9`. On a similar note, we also noticed that
some columns can be obtained by linear transformations of other columns.
`streets_per_node_counts_*` and `streets_per_node_proportion_*` are a
great example of this because one is just a normalized version of the
other.

## Rationale behind the output

At first, we considered creating regression models that output either a
confidence interval or probability distribution. Since the marginal
distribution of `unacast_session_count` is skewed, we thought that these
kinds of estimates would be more robust to outliers than a single-value
prediction. However, given that the end users (i.e. playground owners
and managers) are more comfortable working with single-value predictions
than estimates that incorporate uncertainty, we chose to build models
that predict either the mean or median `unacast_session_count`. The
performance of these models were evaluated using the root mean squared
error (RMSE) and mean absolute error (MAE), respectively. Quantile
regression was also pursued here because the mean is less sensitive to
extreme values than the mean.

## Rationale behind the data split

The dataset consists of 24 monthly observations for 2506 Biba-enabled
playgrounds in the United States. The dates ranged from January 2018 to
December 2019. Data from January 2018 were excluded from our analysis
because many observations are missing the target value for this month,
as shown in Figure 5. Therefore, our training set consisted of
observations from February 2018 through June 2019 and our validation set
included observations from July 2019 through September 2019. The
observations from the last three months were aside for model testing.
This strategy enabled us to avoid data leakage when pursuing a time
series approach.

## Analysis with the old dataset

The data used in this iteration of modeling can be found
[here](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/data/old_train_data.zip).
On Google Drive, it is saved as `playground_stats.csv`. Since the focus
of this iteration was not on preprocessing, rows missing the target
value were dropped and missing values in the explanatory variables were
imputed with zeros.
[`/src/preprocessing_old.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/preprocessing_old.py)
contains the functions that were used to clean the data prior to
modeling.

Nine algorithms were used. Table 1 shows where the .ipynb file for each
algorithm can be found.

Table 1.

|                                                                                                                                          Filename | Algorithms |
| ------------------------------------------------------------------------------------------------------------------------------------------------: | ---------- |
|                     [`/src/training_LGBM_01.ipynb`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_LGBM_01.ipynb) |            |
|   [`/src/training_random_forest_01.ipynb`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_random_forest_01.ipynb) |            |
| [`/src/training_gradient_boost_01.ipynb`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_gradient_boost_01.ipynb) |            |
|     [`/src/training_SVR_CatBoost_01.ipynb`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_SVR_CatBoost_01.ipynb) |            |

Across the board, these rudimentary models performed poorly. The
validation RMSE values were in the range of 300 to 600.

We also fit models to data in which the number of dimensions was reduced
via PCA. PCA was performed in two ways: (1) on the whole dataset and (2)
on groups of related columns.
[`/src/PCA_data.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/PCA_data.py)
contains the functions used to perform PCA. Table 2 describes where the
work can be found.

Table 2.

| Filename | Algorithms |
| -------: | ---------- |
|          |            |
|          |            |
|          |            |
|          |            |

Out of curiousity, we also tried fitting models to data in which the
playgrounds with historic session counts greater than 70,000 were
removed. This dramatically improved the fit of the model, decreasing the
error in both the training and validation set.
[`/src/training_gradient_boost_01.ipynb`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_gradient_boost_01.ipynb)
illustrates the improvement in model performance.

## Analysis with the new dataset

The second iteration of modeling was performed using \[this data\]
(point to the .csv file in the GitHub repository and Google Drive). The
major difference between the new and old dataset is the range of the
target variable. `unacast_session_counts` over 3000 were normalized to
fall between 3000 and 4000. This was done because in the previous
iteration, we saw that removing observations from the super playgrounds
(i.e. the outliers), the model fit improved dramatically.

About 5 different models were pursued during this iteration. The list
below shows which Jupyter Notebook contains work on which algorithm.

  - SVM; random forest; different flavours of gradient boosting

It should be noted that, to speed up computation, these notebooks were
run on Amazon EC2.

In addition to these off-the-shelf regressors, a time-dependent model
was built where the lagged target variable was included as an
explanatory variable. We assumed that session counts would be similar
across consecutive months for a given playground so that these lagged
values could serve as useful input signals. \[Click here to jump to the
Notebook\]

Two other models were built to address the skewness in the target
variable. A mixed effects model was also pursued. The playgrounds were
placed in groups based on a value of a categorical variable or
clustering algorithms such as k-means. The intention was to fit a
different regression surface for each group of playgrounds, instead of
trying to fit a single hyperplane to the data. \[Click here to jump to
the notebook\]

We also considered a tiered approach. This model consisted of a
classifier which would predict an observation to be either low count or
high count. Based on that decision, a prediction would be made using a
regressor that was trained on low-count or high-count data. \[Click here
to jump to the notebook\].

However, none of these models outperformed the boosting models mentioned
earlier.

One that might be worth mentioning is a trend that we observed in many
of the residual plots(include residual plots of different models).

As not all the playgrounds seemed to behave the same we (for example a
playground in South of the US may have more visits in December than in
summer because winter is not very cold, while summer is very hot), but
there where not enough data per playground and to many playgrounds to
run a different model for each playground, we decided to implement mixed
linear models. The way we did it is that we clustered the playgrounds
using some similarities (explained latter) and then the model used fit a
regression hyperplan on the whole data. Once we have this hyperplan, for
each different hyperplan, the model adds a constant to it, so that there
is one regression hyperplan per cluster. By doing it this way, the
regression plan of one cluster depends on the regression plan of another
cluster, but each cluster still have it’s own equation, so the model is
less geenric, more specific to a given playground.

We tried to do it in both R (using `lmer` function) and Python (`smf`
function from the `statsmodels.formula.api` library)

In R,we tried to cluster the data using the `state`, the `climate`, the
`density_class` and the `income_class`. No improvement compared to
before, validation RMSE around 200 and validation MAE around 100. We
also tried to run K-means with 2 clusters and 4 clusters, and use those
clusters when we ran the mixed effects model. In this case, we have to
highlight that a same playground may have values in different months. We
decided to use K-means just to see if the results were significantly
better, and digg more into this problem if this was the case.
Unfortunately, this didn’t improve neither our RMSE (200) nor our MAE
(100), so we didn’t digg more into this.

Python :
<https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/training_mixed_effects_Python.ipynb>
Runing Mixed effects model on Python was more of a trouble, because in
contrary to R, the function in Python doesn’t have an arugment that
allows you to drop the columns that make the algorithm not converging.
First thing we had to do was to write a function that drops all those
columns (see chunk of code starting by \#Find the columns that make the
fit function fail). Then, we fitted mixed effects models using different
clusters (as we did in R). The different clusters that we used were
defined by the `climate`, the `density_class` and the `income_class`
features (all separated in different models). We got no improvement here
either. We got validation RMSE around 200 and validation MAE around 100.
We then tried to capped the values at 0, so that we won’t have any
predicted value that would be negative (as it would make no sense).
However, this didn’t have much of an impact on the RMSE not on the MAE.

More preprocessing was considered during this iteration. Imputation
techniques were reconsidered. For some features, missing values were in
fact synonymous with zeros. For others, it made more sense to replace
missing values with the mean or a specific value (i.e. featured related
to election results in Alaska). Further feature engineering and
selection were performed to reduce model complexity. We dropped columns
in which the proportion of missing values was high (provide examples),
removed correlated features (provide examples), and combined columns
using domain knowledge (provide examples). Here are the .py files that
contain the relevant functions.

## Data product

#### Results

Our data product consists of three boosting models that predict the
median `unacast_session_count`. We selected these models because they
are least worst-performing models we came across in our analysis, they
are relatively fast to train, and the median is less sensitive to
extreme values than the mean, as mentioned earlier.

#### Reproducing the data analysis

Instructions on how to run the makefile to reproduce this report can be
found
[here](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/README.md).
The makefile automates the execution of six scripts. The first script
[`/src/01_split_data.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/01_split_data.py)
splits the raw data into the training and test sets. The second script
[`/src/02_preprocessing.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/02_preprocessing.py)
fits an imputer and one-hot encoder on the training set and saves them
as .joblib files for later use. It also transforms the training and test
sets and saves them as .csv files in the
[`/data`](https://github.com/Z2hMedia/capstone_machine_learning/tree/master/data)
directory. Smaller versions of the training and test sets are also saved
as .csv files to serve as dummy data for testing. It should be noted
that the preprocessing methods used here are identical to those used in
the second iteration of modeling.
[`/src/03_gbr_model.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/03_gbr_model.py),
[`src/04_catboost_model.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/04_catboost_model.py),
and
[`05_lgbm_model.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/src/05_lgbm_model.py)
are the scripts in which modeling take place. The hyperparameters are
hard coded based on the results of the random searches performed in the
second iteration of modeling. In each script, a model is fit and then
used to predict on the training and test sets. Since these models can
predict negative values, the nonsensical predictions are converted to
zero prior to calculating the MAE. Each model is saved as a .joblib file
and its performance metrics are saved as a .csv file in the
[`/results`](https://github.com/Z2hMedia/capstone_machine_learning/tree/master/results)
directory. The last script in the makefile renders this report.

#### Predicting on new data

It is also possible to predict `unacast_session_count` values for an
unseen dataset using the models described above. Instructions on how to
run the makefile to predict on new data can be found
[here](https://github.com/Z2hMedia/capstone_machine_learning/blob/master/README.md).
The data is preprocessed in the same way as described above: the imputer
and one-hot encoder from earlier is loaded to transform the data.
[`07_prediction.py`](https://github.com/Z2hMedia/capstone_machine_learning/blob/sirine/src/07_prediction.py)
outputs a .csv file in the
[`/results`](https://github.com/Z2hMedia/capstone_machine_learning/tree/master/results)
directory. Non-negative predictions from the three models are added as
new columns to the input data.

## Recommendations

#### Outliers in the target

It was demonstrated again and again that outliers in
`unacast_session_count` were detrimental to model fit. In the future,
other statistical or machine learning models that are more robust to
outliers could be pursued. Alternatively, the strategy of fitting
multiple hyperplanes to the data could be considered further. However,
we believe that the most effective strategy for improving model fit is
to reevaluate how the target value is calculated using cell phone
location data. Perhaps the polygon that is drawn around some playgrounds
are ill-shaped, giving the impression that more playground visits took
place than there actually were.

#### Missing values in the target

Prior to modeling, we simply removed rows with missing target values.
However, we stress that these values can be dealt with more elegantly.
For example, if evidence emerges that a playground sees visitors year
round, but is located in an area with poor network coverage, missing
values could be filled by multiple imputation. On the other hand, an
observation missing `unacast_session_count` could be dropped if evidence
shows that the playground was not in operation that month. These
scenarios demonstrate that there is no one-size-fits-all solution.

#### Missing values in explanatory variables

We dropped a handful of columns with a high proportion of missing
values. However, some of those features may in fact be important
predictors of playground usage. If time and resources permit, it may be
worth consulting additional external sources to fill in those values.

#### Feature engineering and selection

This may not improve model fit, but further manipulation of the raw data
may allow for other algorithms, which were initially not used for
reasons related to time complexity, to be considered.

## Conclusion

Although the models are in need of improvement, we hope that our work
has at least brought the organization closer to attaining a reliable
predictor that can be used to inform the decision-making process around
community play spaces. It should also be mentioned that these models
were fit to data collected before the pandemic. Although people are
returning to playgrounds as restrictions ease, these models may only be
reflective of pre-pandemic behaviour. Further model tuning may be
required to incorporate the behavioural changes that took place and are
continuing to take place in society.

## Acknowledgements

We would like to thank Biba Ventures Inc. for sharing their resources
and providing unparalleled support over the course of this project. We
extend our gratitude to our mentor, Vincenzo Coia, and the UBC MDS
program for making this experience possible.
