import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

from Modules.TitanicClean import clean_data, log_transformation, drop_column
from Modules.TitanicPlotting import plot_data
from Modules.TitanicModel import create_model
from Modules.KNNimpute import weighted_hamming, distance_matrix, knn_impute
from Modules.TTestPearson import pearsonscorrTest

try:
    import click
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "click", "--quiet"])
    import click

@click.command()
@click.option('--gridsearch', is_flag=True, help='Perform grid search for best model params.')
def main(gridsearch):
	print ('>>> WELCOME <<<\n')

	# Read in CSV of training data
	train_df = pd.read_csv('Data/train.csv')
	uncleaned_df = train_df

	# Read in CSV of testing data
	test_x_df = pd.read_csv('Data/test.csv')
	test_y_df = pd.read_csv('Data/gender_submission.csv')
	test_df = pd.concat([test_x_df, test_y_df], axis=1)

	#-#-# CLEAN TRAINING DATA #-#-#
	print ('>>> DATA CLEANING <<<\n')

	# Sort Pclass column
	train = train_df.sort_values(['Pclass'], ascending=[1])
	# Reset the index
	train = train.reset_index()
	# Clean the training data
	train = clean_data(train)

	# Transform Age and Fare using a log function
	ageTransformed = log_transformation(train, 'Age')
	train = drop_column(train, 'Age')
	fareTransformed = log_transformation(train, 'Fare')
	train = drop_column(train, 'Fare')
	train_final = pd.concat([train, ageTransformed, fareTransformed], axis=1)

	print ('+++ Training data cleaned +++\n')

	#-#-# CLEAN TESTING DATA #-#-#

	# Sort Pclass column
	test = test_df.sort_values(['Pclass'], ascending=[1])
	# Reset the index
	test = test.reset_index()
	# Clean the training data
	test = clean_data(test)

	# Transform Age and Fare using a log function
	ageTransformed = log_transformation(test, 'Age')
	test = drop_column(test, 'Age')
	fareTransformed = log_transformation(test, 'Fare')
	test = drop_column(test, 'Fare')
	test_final = pd.concat([test, ageTransformed, fareTransformed], axis=1)

	print ('+++ Testing data cleaned +++\n')

	# Features to be used in plotting and model
	selected_columns = ['Pclass','Sex_female','Sex_male','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']
	selected_columns_pearson = ['Sex_female','Age','SibSp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S']
	selected_columns_logistic = ['Pclass','Sex_female','Sex_male','Age','Parch','Fare','Embarked_C','Embarked_S']
	
	#-#-# PLOT DATA #-#-#
	print ('\n>>> DATA VISUALIZATION & ANALYSIS <<<\n')
	plot_data(uncleaned_df, train_final, selected_columns)

	#-#-# PEARSON CORRELATION HYPOTHESIS TEST #-#-#
	print ('\n>>> PERFORM HYPOTHESIS TESTING <<<')
	pearsonscorrTest(train_final, selected_columns)

	#-#-# TRAIN AND TEST #-#-#
	print ('>>> DATA MODEL <<<\n')
	print("**** TRAIN AND TEST SVC WITH ALL VARIABLES ****")
	create_model(train_final, test_final, selected_columns, gridsearch)

	print("**** TRAIN AND TEST SVC WITH VARIABLES SELECTED BY PEARSON CORRELATION ****")
	create_model(train_final, test_final, selected_columns_pearson, gridsearch)

	print("**** TRAIN AND TEST SVC WITH VARIABLES SELECTED BY LOGISTIC REGRESSION ****")
	create_model(train_final, test_final, selected_columns_logistic, gridsearch)
	#-#-# GENERATE ROC PLOT #-#-#
	# print ('\n>>> GENERATING ROC PLOT <<<')
	# train_final = pd.read_csv('Data/TitanicDataClean.csv')
	# test_final = pd.read_csv('Data/TitanicDataTestClean.csv')

	# y_test, y_score, predictions = create_roc_model(train_final, test_final, selected_columns)

	# gen_roc_plot(y_test, y_score, predictions)

	print ('\n>>> GOODBYE <<<')

###########
if __name__ == "__main__":
    main()
else:
    print("The module Titanic.py is intended to be executed to clean and visualize the data.")
###########