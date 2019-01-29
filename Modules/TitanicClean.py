import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from Modules.KNNimpute import weighted_hamming, distance_matrix, knn_impute

# Clean the data
def clean_data (df_):

	# Convert 'Embarked' column into numerical columns
	df = make_dummies(df_, 'Embarked')

	# Impute the 'Embarked' columns using KNN
	embarked_c = impute_col(df, 'Embarked_C', 'mode', 0.8)
	embarked_q = impute_col(df, 'Embarked_Q', 'mode', 0.8)
	embarked_s = impute_col(df, 'Embarked_S', 'mode', 0.8)

	# Impute the Age and Fare columns using KNN
	age = impute_age_by_pclass(df)
	fare = impute_col(df, 'Fare', 'mean', 0.9)

	# Drop the following columns
	cols_to_drop = ['Name', 'Cabin', 'Ticket', 'Age', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Fare']
	df = drop_columns(df, cols_to_drop)

	# Concatenate the imputed dataframes
	final_df = pd.concat([df, age, embarked_c, embarked_q, embarked_s, fare], axis=1)

	# Convert 'Sex' column into numerical columns
	final_df = make_dummies(final_df, 'Sex')

	# Clean the 'Age' column by rounding all of the Age values
	final_df = clean_age(final_df)
	final_df = clean_fare(final_df)

	# Drop the 'PassengerId' column
	final_df = drop_column(final_df, 'PassengerId')

	# Drop the created index via resetting index
	final_df = drop_column(final_df, 'index')

	return final_df

# Transform a column using log
def log_transformation (df_, colname_):
	colLogTransformed = df_[colname_].apply(np.log)
	return colLogTransformed

# Impute Age column by Pclass
def impute_age_by_pclass (df_):
	columns = ['PassengerId','Sex','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']

	# Extract each Pclass to impute it
	df_pclass1 = df_.loc[df_['Pclass'] == 1, columns]
	df_pclass2 = df_.loc[df_['Pclass'] == 2, columns]
	df_pclass3 = df_.loc[df_['Pclass'] == 3, columns]

	# Impute each Pclass separately
	pclass1 = impute_col(df_pclass1, 'Age', 'mean', 1.0)
	pclass2 = impute_col(df_pclass2, 'Age', 'mean', 1.0)
	pclass3 = impute_col(df_pclass3, 'Age', 'mean', 1.0)

	# Concat all Pclass
	frames = [pclass1, pclass2, pclass3]
	age = pd.concat(frames)

	# Return the Age column
	return age

# Impute the specified column using KNN with the specified method (mode or mean) and the threshold value
def impute_col (df_, colname_, method_, threshold_):

	imputed_col = knn_impute(target=df_[colname_], attributes=df_.drop([colname_, 'PassengerId'], 1),
                                   aggregation_method=method_, k_neighbors=10, numeric_distance='euclidean',
                                   categorical_distance='hamming', missing_neighbors_threshold=threshold_)
	return imputed_col

# Drop the specified columns
def drop_columns (df_, collist_):
	df_.drop(columns=collist_, axis=1, inplace=True)
	return df_

# Drop the specified column 
def drop_column (df_, colname_):
	df_.drop(columns=[colname_], axis=1, inplace=True)
	return df_

# Make dummies for the column
def make_dummies (df_, colname_):
	df_ = pd.get_dummies(df_, columns = [colname_])
	return df_

# Clean the age column
def clean_age(df_):
	df_.loc[df_['Age'] < 1.0, 'Age'] = 1.0
	df_['Age'] = df_['Age'].astype(int)
	return df_

# Clean the fare column
def clean_fare(df_):
	df_.loc[df_['Fare'] < 1.0, 'Fare'] = 1.0
	df_['Fare'] = df_['Fare'].astype(int)
	return df_
