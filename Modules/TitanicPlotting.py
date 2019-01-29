import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import seaborn as sns
import scipy
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency

def plot_data(train_df_uncleaned, train_df_cleaned, selected_columns):
	
	# Plot data statistics for uncleaned data
	data_exploration(train_df_uncleaned, "uncleaned data")

	# Plot data statistics for cleaned data
	data_exploration(train_df_cleaned, "cleaned data")

	# Compute Pearson Correlation Coefficient
	cm = compute_corr_coef(train_df_cleaned, selected_columns)

	# Plot heatmap
	plot_heatmap(cm, selected_columns)

def data_exploration(df_, data_):
        print('+++ Descriptive summary of', data_, ' +++')
        print('Train number of rows: ', df_.shape[0])
        print('Train number of columns: ', df_.shape[1])
        print('Train set features: %s' % df_.columns.values)
        
        column_labels=list(df_)
        count=df_.count()
        
        pd.options.display.max_rows = 20
        sns.set(style="darkgrid")
        plt.figure()
        ax = sns.barplot(x=column_labels, y=count, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(),rotation=30, ha="right")
        ax.set_title('Number of records Vs. Features in Dataset - ' + data_)
        ax.set_ylabel('Number of records')
        ax.set_xlabel('Features')
        plt.show()	    

        if (data_ == 'uncleaned data'):
        	print ('\n')

def compute_corr_coef(df_, columns_):
	cm = np.corrcoef(df_[columns_].values.T)
	cm = np.nan_to_num(cm)

	return cm

def plot_heatmap(corrcoef_, columns_):
	plt.figure(figsize=(9,6))
	hm = sns.heatmap(corrcoef_, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=columns_, xticklabels=columns_)
	hm.set_title('Pearson Correlation Heatmap')
	plt.yticks(rotation=1)
	plt.xticks(rotation=30) 
	plt.show()

	