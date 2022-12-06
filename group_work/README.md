# COMP9417 Project - Elo Merchant Category Recommendation

## Dream Team - Group members

|      NAME      | Student ID |
|:--------------:|:----------:|
|   Yifei Yue    |  z5392319  |
|   Zhao Zheng   |  z5297877  |
|  Yuanwei Zhao  |  z5355526  |
|   Jiahui Xie   |  z5341211  |
| Xiandong Cheng |  z5342690  |


## Installation

### Git clone the repository or just use the zip file and setup the environment:
1. `git clone https://github.com/yueyifei0716/COMP9417-Project.git`
2. `cd COMP9417-Project-master`
3. `pip install -r requirements.txt`

### Download the kaggle official dataset
1. Go to https://www.kaggle.com/competitions/elo-merchant-category-recommendation/data
2. Click the black **'Download All'** button below
3. Place the download dataset under the COMP9417-Project-master directory


## Train machine learning models on the dataset

### Data preprocessing and feature engineering of the original dataset
1. Run `python preprocessing.py` to combine the multiple datatables
2. Run `python feature_engineering.py` to generate more features based on the preprocessed dataset
3. Run `python split_train_test.py` to split the training set into a new training set and a testing set
4. Run `python get_train_sample.py` to get a small sample of training set and show they have a similar data distribution, because we will run grid search and TPE hyperparameter optimization on this small sample to save computing time

Since the dataset is relatively large and feature engineering takes a long time,
we put all of the intermediate results of the data processing and the final dataset after processing into Google Drive.

You can download from https://drive.google.com/drive/folders/116nkoVGhT9OImWRn4S4vIwFcJOb2FeKL?usp=sharing and place the downloaded **'preprocess'** folder under the COMP9417-Project-master directory,
**after finishing this we can skip the above 4 steps**.

### Train machine learning models on the processed dataset
1. Train randomforest regressor: `python random_forest.py`
2. Train lighGBM regressor: `python light_gbm.py`
3. Train XGBoost regressor: `python xg_boost.py`
4. Train XGBoost random forest regressor: `python xgb_rf.py`
5. Train Neural Network: `python nn_filter.py`
6. Train voting model fusion: `python voting.py`

All training and prediction results are written into CSV files and 
saved under the result directory, and all evaluation results i.e. RMSE will be displayed in the command line.

The result files with the prefix of 'submission_' are the predictions made on the official testing set. We can submit these files to kaggle and obtain the official score (We have completed the submission and finally discussed it in the report).

### The directory structure should be the same as below
```tree
├── README.md
├── elo-merchant-category-recommendation
├── feature_engineering.py
├── get_train_sample.py
├── models
│   ├── feature_selection.py
│   ├── light_gbm.py
│   ├── nn_filter.py
│   ├── nn_origin.py
│   ├── random_forest.py
│   ├── voting.py
│   ├── xg_boost.py
│   └── xgb_rf.py
├── preprocess
│   ├── new_test.csv
│   ├── new_train.csv
│   ├── new_train_30perc.csv
│   ├── test.csv
│   ├── test_dict.csv
│   ├── test_groupby.csv
│   ├── test_pre.csv
│   ├── train.csv
│   ├── train_dict.csv
│   ├── train_groupby.csv
│   ├── train_pre.csv
│   ├── transaction_d_pre.csv
│   └── transaction_g_pre.csv
├── preprocessing.py
├── requirements.txt
├── result
│   ├── lightgbm_filter.csv
│   ├── lightgbm_wrapper.csv
│   ├── randomforest_filter.csv
│   ├── randomforest_wrapper.csv
│   ├── submission_lightGBM_wrapper.csv
│   ├── submission_randomforest_wrapper.csv
│   ├── submission_voting_avg.csv
│   ├── submission_voting_weight.csv
│   ├── submission_xgboost_wrapper.csv
│   ├── voting_avg.csv
│   ├── voting_weight.csv
│   ├── xgboost_filter.csv
│   └── xgboost_wrapper.csv
└── split_train_test.py
```
