import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, classification_report, confusion_matrix
import math
from Modules.GridSearch import grid_search
from Modules.TitanicClean import clean_data, drop_column, clean_age
from scipy import interp
import matplotlib.pyplot as plt

def create_model(train_df, test_df, selected_columns, yes):

	# Data
	x_train=train_df[selected_columns]
	y_train=train_df['Survived']
	x_test=test_df[selected_columns]
	y_test=test_df['Survived']
	
	# SVC model
	print ("+++ Building model using SVC +++")
	svc=SVC(kernel='linear', C=1)
	svc.fit(x_train,y_train)
	score = svc.score(x_train,y_train)
	print ("\n+++ Mean accuracy of training data +++ ")
	print (score)
	predictions = svc.predict(x_test)

	# Accuracy metrics
	cnf_matrix = confusion_matrix(y_test, predictions)
	print ("\n+++ Confusion Matrix +++")
	print (cnf_matrix)

	# Results
	print ("\n+++ Results from model using test data +++")
	print ("Accuracy = {:.2%}".format(accuracy_score(y_test, predictions)))
	print ("Precision = {:.2%}".format(precision_score(y_test, predictions)))
	print ("Recall = {:.2%}".format(recall_score(y_test, predictions)))

	# Cross-validation scores
	print("\n>>> Starting 5-fold cross validation <<<")
	scores = cross_val_score(svc, x_train, y_train, cv=5)
	print ("\n+++ Cross validation scores +++ ")
	print(scores)

	# Mean score and 95% confidence interval of the score
	print("Accuracy of base SVC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
	SE = scores.std()/math.sqrt(5)
	print("95%% CI: [%0.4f, %0.4f]" % (scores.mean() - 1.96*SE, scores.mean() + 1.96*SE))

	#ROC analysis
	print ('\n>>> GENERATING ROC PLOT <<<')
	cv = StratifiedKFold(n_splits=6)

	tprs = []
	aucs = []
	mean_fpr = np.linspace(0,1,100)

	plt.figure()
	i=0
	X = x_train.as_matrix()
	y = y_train.as_matrix()
	svc2 = SVC(kernel='linear', probability=True)
	for train, test in cv.split(X, y):
		probas_ = svc2.fit(X[train], y[train]).predict_proba(X[test])
		#probas_ = svc.predict(X[test]).
		fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1])
		tprs.append(interp(mean_fpr, fpr, tpr))
		tprs[-1][0] = 0.0
		roc_auc = auc(fpr, tpr)
		aucs.append(roc_auc)
		plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

		i+=1
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

	mean_tpr = np.mean(tprs, axis=0)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	std_auc = np.std(aucs)
	plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

	std_tpr = np.std(tprs, axis=0)
	tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
	tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
	plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

	if yes:
		# Grid Search
		print("\n\n>>> Starting Grid Search... <<<")
		new_svc = grid_search(svc, x_train, y_train)
		print("Best parameters set found on development set:")
		print()
		print(new_svc.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = new_svc.cv_results_['mean_test_score']
		stds = new_svc.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, new_svc.cv_results_['params']):
			se = std / math.sqrt(5)
			print("%0.3f (+/-%0.03f) with 95%% CI [%0.3f, %0.3f] for %r" % (
				mean, std, mean - 1.96 * se, mean + 1.96 * se, params))
		print()

		new_prediction = new_svc.predict(x_test)
		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		print(classification_report(y_test, new_prediction))
		print(confusion_matrix(y_test, new_prediction))

		# Results
		print("Accuracy = {:.2%}".format(accuracy_score(y_test, new_prediction)))
		print("Precision = {:.2%}".format(precision_score(y_test, new_prediction)))
		print("Recall = {:.2%}".format(recall_score(y_test, new_prediction)))