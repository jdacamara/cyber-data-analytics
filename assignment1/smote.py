import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def prepare_and_split_data():
	data = pd.read_csv('data_for_student_case.csv')
	data = data[data.simple_journal != 'Refused']
	
	#Changes the simple_journal value to binary
	data['simple_journal'].replace(['Chargeback'], 1, inplace=True)
	data['simple_journal'].replace(['Settled'], 0, inplace=True)

	#Sort the important columns
	model_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode','simple_journal', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode' ]
	relevent_data = data[model_variables]

	#Change the value to one hot-encoding
	relevant_data_enconded = pd.get_dummies(relevent_data)

	#print relevant_data_enconded

	#split the data
	training_features, test_features, \
	training_target, test_target, = train_test_split(relevant_data_enconded.drop(['simple_journal'], axis=1),
			                                           relevant_data_enconded['simple_journal'],
			                                           test_size = .1)


	x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
	                                                  test_size = .1)

	return x_train, x_val, y_train, y_val



def smote_testing_for_classifier(classifier, label):
	#load data and remove refused data
	data = pd.read_csv('data_for_student_case.csv')
	data = data[data.simple_journal != 'Refused']
	
	#Changes the simple_journal value to binary
	data['simple_journal'].replace(['Chargeback'], 1, inplace=True)
	data['simple_journal'].replace(['Settled'], 0, inplace=True)

	#Sort the important columns
	model_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode','simple_journal', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode' ]
	relevent_data = data[model_variables]

	#Change the value to one hot-encoding
	relevant_data_enconded = pd.get_dummies(relevent_data)

	#print relevant_data_enconded

	#split the data
	training_features, test_features, \
	training_target, test_target, = train_test_split(relevant_data_enconded.drop(['simple_journal'], axis=1),
			                                           relevant_data_enconded['simple_journal'],
			                                           test_size = .1)


	x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
	                                                  test_size = .1)

	"""
	sm = SMOTE(random_state=12, ratio = 0.1)
	x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


	clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
	clf_rf.fit(x_train_res, y_train_res)


	print 'Validation Results'
	print clf_rf.score(x_val, y_val)
	print recall_score(y_val, clf_rf.predict(x_val))
	print '\nTest Results'
	print clf_rf.score(test_features, test_target)
	print recall_score(test_target, clf_rf.predict(test_features))

	"""

	#model_LR= LogisticRegression()
	classifier.fit(x_train,y_train)

	plt.figure(figsize=(10,10))
	plt.title('Receiver Operating Characteristic for %s' %label)
	
	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	roc_auc = auc(false_positive_rate, true_positive_rate)


	plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
	


	sm = SMOTE(ratio=0.4)
	x_train10, y_train10 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train,y_train)
	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	roc_auc = auc(false_positive_rate, true_positive_rate)

	plt.plot(false_positive_rate,true_positive_rate)

	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],linestyle='--')
	plt.axis('tight')
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.legend(loc="lower right")
	plt.show()

def smote_testing_for_class(classifier, label):
	x_train, x_val, y_train, y_val = prepare_and_split_data()

	classifier.fit(x_train, y_train).decision_function(x_val)
	y_score_no_sampling = classifier.decision_function(x_val)
	y_pred_no_sampling = classifier.predict(x_val)

	precision = dict()
	recall = dict()
	average_precision = dict()
	precision, recall, thresholds = precision_recall_curve(y_test,y_score_no_sampling)
	average_precision = average_precision_score(y_test,y_score_no_sampling)

	