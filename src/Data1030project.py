#######################
### import packages ###
#######################
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


###################
### engineering ###
###################
data = pd.read_csv('/Users/ciciwxp/Desktop/DATA1030/project/kindle_data-v2.csv')
data = pd.DataFrame(data)

## drop useless columns
drop_columns = ['asin', 'author', 'title', 'imgUrl', 'productURL', 'category_id']
data = data.drop(columns = drop_columns, axis = 1)

## drop value Y = null
data = data[data['stars'] != 0]

## transform binary
data = data.replace({True: 1, False: 0})

## add transformed date column
date_format = "%Y-%m-%d"
data['publishedDate_clean'] = data['publishedDate'].apply(lambda x: 
    datetime.strptime(str(x), '%Y-%m-%d') if not pd.isna(x) else x)
origin = data['publishedDate_clean'].min()
# data['publishedDate_num'] = pd.to_numeric(data['publishedDate_clean'].apply(lambda x: (x - origin).days if not pd.isna(x) else x), errors='coerce') 
data['published_month'] = data['publishedDate_clean'].dt.month
data['published_year'] = data['publishedDate_clean'].dt.year
data['published_year'] = data['published_year'][data['published_year'].notna()].astype(int)
data['published_days'] = data['publishedDate_clean'].dt.day_name()

data.columns
## drop target variable
Y = data['stars']
X = data.drop(columns = ['stars', 'publishedDate', 'publishedDate_clean'], axis = 1)

## count NA
missing_values = X.isna().sum()
48232/len(X)

###########
### EDA ###
###########

data.corr()

## histogram for target variable
tb_isKindleUnlimited = data[['stars', 'isKindleUnlimited']]
tb_isKindleUnlimited1 = tb_isKindleUnlimited[tb_isKindleUnlimited['isKindleUnlimited'] == 1]
tb_isKindleUnlimited0 = tb_isKindleUnlimited[tb_isKindleUnlimited['isKindleUnlimited'] == 0]
hist_params = {
    'bins': 32,
    'alpha': 0.7,
    'edgecolor': 'black',
}

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms for Kindle Unlimited and non-Kindle Unlimited books
ax.hist(tb_isKindleUnlimited0['stars'], color='#146eb4', label='Not Kindle Unlimited', **hist_params)
ax.hist(tb_isKindleUnlimited1['stars'], color='#FF9900', label='Kindle Unlimited', **hist_params)

# Calculate mean values for both groups
mean_not_kindle_unlimited = np.mean(tb_isKindleUnlimited0['stars'])
mean_kindle_unlimited = np.mean(tb_isKindleUnlimited1['stars'])

# Plot mean lines
ax.axvline(mean_not_kindle_unlimited, color='blue', linestyle='--', label=f'Mean (Not Kindle Unlimited): {mean_not_kindle_unlimited:.2f}')
ax.axvline(mean_kindle_unlimited, color='orange', linestyle='--', label=f'Mean (Kindle Unlimited): {mean_kindle_unlimited:.2f}')

# Set labels and title
ax.set_xlabel('Rating [Stars] by Kindle Unlimited')
ax.set_ylabel('Number of Books')
ax.set_title('Distribution of Kindle Book Ratings by Kindle Unlimited')

# Add a legend
ax.legend()

# Customize the grid and axis
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.show()

## overall star hist
plt.figure(figsize=(10, 6))
plt.hist(data['stars'], color='#FF9900', **hist_params)
plt.ylabel('Counts')
plt.xlabel('Ratings [stars]')
plt.title('Target: Ratings [stars]')
plt.show()



## predictors: binary variable () in pie chart and box plot
# "isBestSeller"
tb_isBestSeller = data[['stars', 'isBestSeller']]
tb_isBestSeller.boxplot(by = 'isBestSeller', showfliers=True, 
                         flierprops=dict(marker='o', markersize=1, markerfacecolor='#ffc34e'))


# "isEditorsPick"
tb_isEditorsPick = data[['stars', 'isEditorsPick']]
tb_isEditorsPick.boxplot(by = 'isEditorsPick', showfliers=True, 
                         flierprops=dict(marker='o', markersize=1, markerfacecolor='#ffc34e'))


# "isKindleUnlimited"
tb_isKindleUnlimited.boxplot(by = 'isKindleUnlimited', showfliers=True, 
                         flierprops=dict(marker='o', markersize=1, markerfacecolor='#ffc34e'))


# "isGoodReadsChoice"
tb_isGoodReadsChoice = data[['stars', 'isGoodReadsChoice']]
tb_isGoodReadsChoice.boxplot(by = 'isGoodReadsChoice', showfliers=True, 
                         flierprops=dict(marker='o', markersize=1, markerfacecolor='#ffc34e'))


# "category_name"
tb_category_name = data[['stars', 'category_name']]
tb_category_name.boxplot(by = 'category_name', showfliers=True, 
                         flierprops=dict(marker='o', markersize=1, markerfacecolor='#ffc34e'))
plt.xticks(rotation = 90)
plt.xlabel('category')




# “soldBy”
## engineer soldBy: only leave publisher with 1000+ obs, change other publishers to 'other', add changed column 'soldBy_2' to data
## 'soldBy_2' has 13 values: the top 12 publishers and other
soldBy_clean = data['soldBy'].value_counts()[:12].index
data['soldBy_2'] = data['soldBy'].apply(lambda x: 'other' if x not in soldBy_clean else x)
soldBy_order = soldBy_clean.tolist() + ['other']

data['soldBy_2'] = pd.Categorical(data['soldBy_2'], categories=soldBy_order, ordered=True)

# Create the box plot using Seaborn
plt.figure(figsize=(10, 6))

# Create the box plot and store the axis
ax = sns.boxplot(x='stars', y='soldBy_2', data=data, order=soldBy_order, orient='h', showfliers=True,
                 flierprops=dict(marker='o', markersize=5, markerfacecolor='#ffc34e', alpha=0.1))

# Customize the y-axis ticks
ax.yaxis.tick_right()  # Move y-ticks to the right
ax.yaxis.set_label_position("right")  # Move y-label to the right
ax.yaxis.set_label_coords(1.45, 0.5)
ax.set_ylabel('Top 12 Publisher and other', fontsize=12)  # Y-label for the right side
ax.yaxis.label.set_rotation(270)
# Set x-label and title
ax.set_xlabel('Ratings [Stars]', fontsize=12)  # X-label for the left side
plt.title('Ratings by Publishers', fontsize=16)
plt.show()



category_name_clean = data['category_name'].value_counts()[:12].index
data['category_name_2'] = data['category_name'].apply(lambda x: 'other' if x not in category_name_clean else x)
category_name_order = category_name_clean.tolist() + ['other']

data['category_name_2'] = pd.Categorical(data['category_name_2'], categories=category_name_order, ordered=True)

# Create the box plot using Seaborn
plt.figure(figsize=(10, 6))

# Create the box plot and store the axis
ax = sns.boxplot(x='stars', y='category_name_2', data=data, order=category_name_order, orient='h', showfliers=True,
                 flierprops=dict(marker='o', markersize=5, markerfacecolor='#ffc34e', alpha=0.1))

# Customize the y-axis ticks
ax.yaxis.tick_right()  # Move y-ticks to the right
ax.yaxis.set_label_position("right")  # Move y-label to the right
ax.yaxis.set_label_coords(1.35, 0.5)
ax.set_ylabel('Top 12 Categories and other', fontsize=12)  # Y-label for the right side
ax.yaxis.label.set_rotation(270)
# Set x-label and title
ax.set_xlabel('Ratings [Stars]', fontsize=12)  # X-label for the left side
plt.title('Ratings by Categories', fontsize=16)
plt.show()




plt.figure(figsize=(12, 6))

# “reviews”
plt.subplot(1, 2, 1)
tb_reviews = data[['stars', 'reviews']]
tb_reviews = tb_reviews[tb_reviews['reviews'] <= 40000]
plt.scatter(tb_reviews['reviews'], tb_reviews['stars'], s = 2, color = '#1558ed', alpha = 0.3)
plt.ylabel('Ratings [stars]')
plt.xlabel('Number of Reviews')
plt.title('Number of Reviews VS. Ratings [stars]')
x = data['reviews']
y = data['stars']
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)


# Plot the line of best fit
plt.plot(x, line(x), color='red', linewidth=2, label='Line of Best Fit')
#plt.legend(loc='lower right')
#plt.show()



# “price”
plt.subplot(1, 2, 2) 
tb_price = data[['stars', 'price']]
#tb_price = tb_price[tb_price['price'] <= 400]
plt.figure(figsize=(6,6))
plt.scatter(tb_price['price'], tb_price['stars'], s=2, color='#ffc34e', alpha=0.3)
plt.ylabel('Ratings [stars]')
plt.xlabel('Price ($)')
plt.title('Book Price VS. Ratings')

# Calculate the line of best fit
x = tb_price['price']
y = data['stars']
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)

# Plot the line of best fit
plt.plot(x, line(x), color='red', linewidth=2, label='Line of Best Fit')
plt.legend(loc='lower right')

plt.show()


# "publishedDate_num"
tb_date = data[['stars', 'publishedDate_num']]
plt.scatter(tb_date['publishedDate_num'], tb_date['stars'], s=2, color='#244674', alpha=0.1)
plt.ylabel('Stars')
plt.xlabel('publishedDate_num')
plt.title('publishedDate_num VS. Stars')

# Calculate the line of best fit
x = tb_date['publishedDate_num']
y = tb_date['stars']
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)

# Plot the line of best fit
plt.plot(x, line(x), color='red', linewidth=2)

plt.show()


tb_binary = data[['isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'isKindleUnlimited']]

from scipy.stats import chi2_contingency
def cramers_V(var1,var2):
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))
import itertools
for col1, col2 in itertools.combinations(tb_binary.columns, 2):
    print(col1, col2, cramers_V(tb_binary[col1], tb_binary[col2]))

corr_matrix = tb_binary.corr()

# Set the figure size
plt.figure(figsize=(10, 8))

# Create a more visually appealing heatmap using Seaborn
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', vmin=-1, vmax=1, cbar=True, square=True)

# Set labels and title
plt.title('Correlation Matrix Heatmap for 4 Binary Variables', fontsize=14)
plt.xticks(np.arange(4) + 0.5, tb_binary.columns, rotation=0)
plt.yticks(np.arange(4) + 0.5, tb_binary.columns, rotation=0)

# Add grid lines
plt.grid(True, linewidth=0.5, color='white', linestyle='--')

# Customize the color bar
cbar = plt.colorbar()
cbar.set_label('Correlation', rotation=90)

plt.show()




##################
### preprocess ###
##################

# split test and other set
stars_class = round(Y)
#X_other, X_test, Y_other, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42, stratify=stars_class)
#print(f"  Test:{X_test.index}")

# plt.hist(Y_test, bins= 32, color = '#146eb4')
# plt.ylabel('Number of Books')
# plt.xlabel('Reviews [Stars]')
# plt.title('Y_test distribution')

# plt.hist(Y_other, bins= 32, color = '#FF9900')
# plt.ylabel('Number of Books')
# plt.xlabel('Reviews [Stars]')
# plt.title('Y_other distribution')


# replace nan with impossible value for ordinal encoder
# X['published_year'] = X['published_year'].replace(np.NaN, 111111)
# X['published_month'] = X['published_month'].replace(np.NaN, 111111)
# X['published_days'] = X['published_days'].replace(np.NaN, '111111')
# unique_year = X['published_year'].unique()
# unique_year.sort()
# unique_month = X['published_month'].unique()
# unique_month.sort()
# unique_days = X['published_days'].unique()


unique_year = X['published_year'].unique()
unique_year.sort()
unique_year = unique_year[:-1]



# encoder
ordinal_ftrs = ['published_month', 'published_days', 'published_year'] 
ordinal_cats = [[1,2,3,4,5,6,7,8,9,10,11,12], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday'], unique_year] 
onehot_ftrs = ['soldBy', 'category_name','isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'isKindleUnlimited']
std_ftrs = ['reviews', 'price']
numeric_imputer = SimpleImputer(strategy='median')  # or 'mean'
categorical_imputer = SimpleImputer(strategy='most_frequent')  # or 'constant', fill_value='missing'


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), std_ftrs),
        ('ord', Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('encode', OrdinalEncoder(categories=ordinal_cats))]), ordinal_ftrs),
        ('onehot', Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))]), onehot_ftrs)
    ]
)

clf = Pipeline(steps=[('preprocessor', preprocessor)])

preprocessor2 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('scale', StandardScaler())]), std_ftrs),
        ('ord', Pipeline(steps=[('encode', OrdinalEncoder(categories=ordinal_cats))]), ordinal_ftrs),
        ('onehot', Pipeline(steps=[('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))]), onehot_ftrs)
    ]
)

clf2 = Pipeline(steps=[('preprocessor', preprocessor2)])

bins = [0, 4.0, 4.5, 5.1]
labels = ['Low', 'Medium', 'High']
Y_binned = pd.cut(Y, bins=bins, labels=labels, include_lowest=True)
sum(Y_binned.isna())
Y_binned.shape
# stratified K fold


# feature_names = preprocessor.get_feature_names_out()


## engineer soldBy: only leave publisher with 1000+ obs, change other publishers to 'other', add changed column 'soldBy_2' to data
## 'soldBy_2' has 13 values: the top 12 publishers and other
#soldBy_clean = data['soldBy'].value_counts()[:12].index
#data['soldBy_2'] = data['soldBy'].apply(lambda x: 'other' if x not in soldBy_clean else x)
#print(data['category_name'].value_counts())

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
fixed_seed = 0

# Adjusting the baseline calculation for 'stars' column

# Calculate the mean stars (baseline prediction for 'stars')
mean_stars = data['stars'].mean()

# Create a list of baseline predictions for 'stars'
baseline_predictions_stars = [mean_stars] * len(data)

# Actual star ratings
actual_stars = data['stars'].values

# Calculate RMSE for the 'stars' column
baseline_rmse_stars = mean_squared_error(actual_stars, baseline_predictions_stars, squared=False)
baseline_rmse_stars


def MLpipe_KFold_RMSE_Stratified(X, y, y_binned, preprocessor, ML_algo, param_grid):
    test_scores = []
    best_models = []
    r2_scores = []

    for state in range(3):
        print(f"\nRandom State: {state}")

        # Splitting the data
        X_other, X_test, Y_other, Y_test = train_test_split(X, y, test_size=0.2, random_state=state, stratify=y_binned)
        print("Length of X_other:", len(X_other))
        print("Length of Y_other:", len(Y_other))
        
        y_binned_other = pd.cut(Y_other, bins=bins, labels=labels, include_lowest=True)

        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=state)

        # Initialize best_score and best_model for each random state
        best_score = 1000000
        best_model = None

        for train_index, val_index in skf.split(X_other, y_binned_other):
            X_train, X_val = X_other.iloc[train_index], X_other.iloc[val_index]
            Y_train, Y_val = Y_other.iloc[train_index], Y_other.iloc[val_index]

            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('ML_algo', ML_algo)])
            grid = GridSearchCV(pipe, param_grid, scoring=make_scorer(mean_squared_error, greater_is_better=False))
            grid.fit(X_train, Y_train)

            model = grid.best_estimator_
            Y_pred = model.predict(X_val)
            rmse = mean_squared_error(Y_val, Y_pred, squared=False)

            if rmse < best_score:
                best_score = rmse
                best_model = model

        # Predicting and scoring on the test set
        Y_pred_test = best_model.predict(X_test)
        test_rmse = mean_squared_error(Y_test, Y_pred_test, squared=False)
        test_r2 = r2_score(Y_test, Y_pred_test)

        test_scores.append(test_rmse)
        best_models.append(best_model)
        r2_scores.append(test_r2)

        print(f"Model Name: {ML_algo.__class__.__name__}")
        print(f"Best Model: {best_model}")
        print(f"Test RMSE: {test_rmse}")
        print(f"Test R2: {test_r2}")

    return test_scores, r2_scores, best_models



## model parameters:
model_param = {
    'Lasso': (Lasso(random_state=fixed_seed), {'ML_algo__alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}),
    'Ridge': (Ridge(random_state=fixed_seed), {'ML_algo__alpha': [0.01, 0.1, 1, 10, 100, 1000]}),
    'ElasticNet': (ElasticNet(random_state=fixed_seed), {'ML_algo__alpha':  [0.0001, 0.001, 0.01, 0.1, 1, 10], 'ML_algo__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]}),
    'KNN': (KNeighborsRegressor(), {'ML_algo__n_neighbors': [3, 5, 7, 10]}),
    'SVR': (SVR(), {'ML_algo__C': [0.1, 1, 10], 'ML_algo__gamma': ['scale', 'auto']}),
    'RandomForest': (RandomForestRegressor(random_state=fixed_seed), {'ML_algo__n_estimators': [200, 300], 'ML_algo__max_depth': [10], 'ML_algo__max_features': [0.5,0.75,1.0]}),
    'XGBoost': (XGBRegressor(), {
    'ML_algo__max_depth': [3, 4, 5],
    'ML_algo__learning_rate': [0.1, 0.01, 0.001],
    'ML_algo__n_estimators': [10000]
})
}
test_score_knn_stratified, r2_score_knn_stratified, best_models_knn_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['KNN'][0], model_param['KNN'][1])


# LInear models
test_score_lasso_stratified, r2_score_lasso_stratified, best_models_lasso_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['Lasso'][0], model_param['Lasso'][1])
test_score_lasso_mean = pd.Series(test_score_lasso_stratified).mean()
test_score_lasso_sd = pd.Series(test_score_lasso_stratified).std()
test_score_ridge_stratified, r2_score_ridge_stratified, best_models_ridge_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['Ridge'][0], model_param['Ridge'][1])
test_score_ridge_mean = pd.Series(test_score_ridge_stratified).mean()
test_score_ridge_sd = pd.Series(test_score_ridge_stratified).std()
test_score_elastic_stratified, r2_score_elastic_stratified, best_models_elastic_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['ElasticNet'][0], model_param['ElasticNet'][1])
test_score_elastic_mean = pd.Series(test_score_elastic_stratified).mean()
test_score_elastic_sd = pd.Series(test_score_elastic_stratified).std()

## No need for elastic because the model is already underfit, where alpha value and l1 regulation are at smallest: 0.0001, 0

# Random Forest
# fit 1: ML_algo__n_estimators': [10, 50, 100, 200], 'ML_algo__max_depth': [3, 5, 10], BEST:10, 200
# this is the best
test_score_rf_stratified, r2_score_rf_stratified, best_models_rf_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['RandomForest'][0], model_param['RandomForest'][1])
test_score_rf_mean = pd.Series(test_score_rf_stratified).mean()
test_score_rf_sd = pd.Series(test_score_rf_stratified).std()
test_score_rf_stratified2, r2_score_rf_stratified2, best_models_rf_stratified2 = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['RandomForest'][0], model_param['RandomForest'][1])

#{'ML_algo__n_estimators': [200, 300], 'ML_algo__max_depth': [10], 'ML_algo__max_features': [0.5,0.75,1.0]}
test_score_rf_stratified3, r2_score_rf_stratified3, best_models_rf_stratified3 = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['RandomForest'][0], model_param['RandomForest'][1])

best_models_rf_stratified
## XGBoost  
test_score_xgb_stratified, r2_score_xgb_stratified, best_models_xgb_stratified = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, model_param['XGBoost'][0], model_param['XGBoost'][1])
test_score_xgb_mean = pd.Series(test_score_xgb_stratified).mean()
test_score_xgb_sd = pd.Series(test_score_xgb_stratified).std()
test_score_xgb_stratified2, r2_score_xgb_stratified2, best_models_xgb_stratified2 = MLpipe_KFold_RMSE_Stratified(X, Y, Y_binned, clf, XGBRegressor(), {
    'ML_algo__max_depth': [2, 3],
    'ML_algo__learning_rate': [0.01, 0.1],
    'ML_algo__n_estimators': [10000],
    'ML_algo__alpha': [0.0001, 0.001]
})
best_models_xgb_stratified[0].get_params()

## MAX-DEPTH BEST IS STILL 3

best_models_xgb_stratified
# Modifying the MLpipe_KFold_RMSE function to ensure correct stratification

model_names = ['Baseline', 'Lasso', 'Ridge', 'Elastic Net', 'Random Forest', 'XGBoost']
means = [baseline_rmse_stars, test_score_lasso_mean, test_score_ridge_mean, test_score_elastic_mean, test_score_rf_mean, test_score_xgb_mean]
std_devs = [0, test_score_lasso_sd, test_score_ridge_sd, test_score_elastic_sd, test_score_rf_sd, test_score_xgb_sd]  # Dummy standard deviation values for models

# Create the plot
plt.figure(figsize=(10, 6))
plt.errorbar(model_names, means, yerr=std_devs, fmt='o', ecolor='gray', elinewidth=3, capsize=0)
plt.title('RMSE Mean and Standard Deviation')
plt.xlabel('Models')
plt.ylabel('RMSE Mean Values')
plt.grid(True)
plt.show()




################# feature importance ##################
from joblib import load
xgb_model = load('/Users/ciciwxp/Desktop/xgb_model_1206.joblib')
from sklearn.inspection import permutation_importance

X_other, X_test, Y_other, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y_binned)
        
y_binned_other = pd.cut(Y_other, bins=bins, labels=labels, include_lowest=True)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
for train_index, val_index in skf.split(X_other, y_binned_other):
    X_train, X_val = X_other.iloc[train_index], X_other.iloc[val_index]
    Y_train, Y_val = Y_other.iloc[train_index], Y_other.iloc[val_index]
    
X_prep = preprocessor.fit_transform(X_train)
feature_names = preprocessor.get_feature_names_out()

df_train = pd.DataFrame(data=X_prep,columns=feature_names)
df_CV = preprocessor.transform(X_val)
df_CV = pd.DataFrame(data=df_CV,columns = feature_names)
df_test = preprocessor.transform(X_test)
df_test = pd.DataFrame(data=df_test,columns = feature_names)

perm_importance = permutation_importance(xgb_model, df_test, Y_test, n_repeats=30, random_state=0)
sorted_idx = perm_importance.importances_mean.argsort()[-10:]

top_features = df_test.columns[sorted_idx]
top_importances = perm_importance.importances_mean[sorted_idx]
top_std = perm_importance.importances_std[sorted_idx]

plt.barh(range(len(top_importances)), top_importances, xerr=top_std)
plt.yticks(range(len(top_importances)), [top_features[i] for i in range(len(top_importances))])
plt.xlabel('Permutation Importance')
plt.title('Permutation Importance Top 10 Feature Importances')
plt.show()

fig, ax = plt.subplots()
plt.figure(figsize=(10, 6))
ax.boxplot(perm_importance.importances[sorted_idx].T,
           vert=False, labels=top_features)
plt.axvline(x=0, color='grey', linestyle='--')
plt.xlabel('Permutation Importance')
ax.set_title('Permutation Importances (test set)')
plt.show()

#WEIGHT
xgb_model = best_models_xgb_stratified[0].named_steps['ML_algo']
feature_importances_weight = xgb_model.get_booster().get_score(importance_type='weight')
mapped_feature_importances = {feature_names[int(k[1:])]: v for k, v in feature_importances_weight.items()}
sorted_features_weight = sorted(mapped_feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

features_weight, scores_weight = zip(*sorted_features_weight)

plt.barh(range(len(scores_weight)), scores_weight)
plt.yticks(range(len(scores_weight)), features_weight)
plt.xlabel('Weight')
plt.title('Weight - Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()


from joblib import dump

# Assume 'model' is your trained scikit-learn model
dump(xgb_model, '/Users/ciciwxp/Desktop/xgb_model_1206.joblib')

import numpy
import shap

xgb_model = load('/Users/ciciwxp/Desktop/xgb_model_1206.joblib')
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(df_test)

shap.summary_plot(shap_values, df_test, plot_type="bar", max_display=10)

indices = [0, 50, 100]
df_test = preprocessor.transform(X_test)
for index in indices:
    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value[0]
    shap.force_plot(expected_value, shap_values[index], features=df_test[index, :], feature_names=feature_names, matplotlib=True)


y_pred = xgb_model.predict(df_test)
