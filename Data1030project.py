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
import matplotlib.pyplot as plt

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





# “reviews”
tb_reviews = data[['stars', 'reviews']]
plt.scatter(tb_reviews['reviews'], tb_reviews['stars'], s = 2, color = '#1558ed', alpha = 0.1)
plt.ylabel('Stars')
plt.xlabel('Number of Reviews')
plt.title('Number of Reviews VS. Stars')
x = tb_reviews['reviews']
y = tb_reviews['stars']
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)

# Plot the line of best fit
plt.plot(x, line(x), color='red', linewidth=2)
plt.show()



# “price”
tb_price = data[['stars', 'price']]
tb_price = tb_price[tb_price['price'] <= 400]
plt.figure(figsize=(10, 6))
plt.scatter(tb_price['price'], tb_price['stars'], s=2, color='#ffc34e', alpha=0.2)
plt.ylabel('Ratings (Stars)')
plt.xlabel('Price ($)')
plt.title('Book Price VS. Ratings')

# Calculate the line of best fit
x = data['price']
y = data['stars']
coefficients = np.polyfit(x, y, 1)
line = np.poly1d(coefficients)

# Plot the line of best fit
plt.plot(x, line(x), color='red', linewidth=2)

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
X_other, X_test, Y_other, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42, stratify=stars_class)
print(f"  Test:{X_test.index}")

plt.hist(Y_test, bins= 32, color = '#146eb4')
plt.ylabel('Number of Books')
plt.xlabel('Reviews [Stars]')
plt.title('Y_test distribution')

plt.hist(Y_other, bins= 32, color = '#FF9900')
plt.ylabel('Number of Books')
plt.xlabel('Reviews [Stars]')
plt.title('Y_other distribution')


# replace nan with impossible value for ordinal encoder
X_other['published_year'] = X_other['published_year'].replace(np.NaN, 111111)
X_other['published_month'] = X_other['published_month'].replace(np.NaN, 111111)
X_other['published_days'] = X_other['published_days'].replace(np.NaN, '111111')
unique_year = X_other['published_year'].unique()
unique_year.sort()
unique_month = X_other['published_month'].unique()
unique_month.sort()
unique_days = X_other['published_days'].unique()


# encoder
ordinal_ftrs = ['published_month', 'published_days', 'published_year'] 
ordinal_cats = [[1,2,3,4,5,6,7,8,9,10,11,12, 111111], ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday', '111111'], unique_year] 
onehot_ftrs = ['soldBy', 'category_name']
std_ftrs = ['reviews', 'price']

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories = ordinal_cats), ordinal_ftrs), 
        ('onehot', OneHotEncoder(sparse=False,handle_unknown='ignore'), onehot_ftrs), 
        ('std', StandardScaler(), std_ftrs)])

clf = Pipeline(steps=[('preprocessor', preprocessor)])

# stratified K fold
X_other = X_other.replace('NaTType', float('nan'))
star_class_other = pd.Series(round(Y_other))
star_class_other.value_counts()
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

for i, (i_train,i_val) in enumerate(skf.split(X_other, star_class_other)):
    X_train, Y_train = X_other.iloc[i_train], Y_other.iloc[i_train]
    X_val, Y_val = X_other.iloc[i_val], Y_other.iloc[i_val]
    
    X_train_binary = X_train[['isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'isKindleUnlimited']]
    X_train_binary.reset_index(drop=True, inplace=True)
    
    X_val_binary = X_val[['isBestSeller', 'isEditorsPick', 'isGoodReadsChoice', 'isKindleUnlimited']]
    X_val_binary.reset_index(drop=True, inplace=True)
    
    X_train_prep = pd.DataFrame(clf.fit_transform(X_train))
    X_train_prep.columns = preprocessor.get_feature_names_out()
    X_train_prep_full = pd.concat([X_train_prep, X_train_binary], axis=1)
    
    X_val_prep = pd.DataFrame(clf.transform(X_val))
    X_val_prep.columns = preprocessor.get_feature_names_out()
    X_val_prep_full = pd.concat([X_val_prep, X_val_binary], axis=1)


feature_names = preprocessor.get_feature_names_out()

X_train_prep_full.shape
X_val_prep_full.shape

## engineer soldBy: only leave publisher with 1000+ obs, change other publishers to 'other', add changed column 'soldBy_2' to data
## 'soldBy_2' has 13 values: the top 12 publishers and other
#soldBy_clean = data['soldBy'].value_counts()[:12].index
#data['soldBy_2'] = data['soldBy'].apply(lambda x: 'other' if x not in soldBy_clean else x)
#print(data['category_name'].value_counts())






