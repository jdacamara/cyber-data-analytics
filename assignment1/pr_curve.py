import pandas as pd
import matplotlib.pyplot as plt
import pickle

from imblearn.over_sampling import SMOTE 
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc

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


def smote_testing_for_classifier_PR_curve(classifier,label):  
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

	#split the data
	training_features, test_features, \
	training_target, test_target, = train_test_split(relevant_data_enconded.drop(['simple_journal'], axis=1),
			                                           relevant_data_enconded['simple_journal'],
			                                           test_size = .1)


	x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
	                                                  test_size = .1)

	classifier.fit(x_train, y_train)

	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'Smote ratio = 0.0')
	
	sm = SMOTE(ratio=0.1)
	x_train10, y_train10 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train10, y_train10)
	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	plt.plot(false_positive_rate,true_positive_rate, color='green',label = 'Smote ratio = 0.1')

	sm = SMOTE(ratio=0.2)
	x_train20, y_train20 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train20, y_train20)
	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	plt.plot(false_positive_rate,true_positive_rate, color='blue',label = 'Smote ratio = 0.2')

	sm = SMOTE(ratio=0.3)
	x_train30, y_train30 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train30, y_train30)
	y_prob = classifier.predict_proba(x_val)[:,1]
	false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, y_prob)
	plt.plot(false_positive_rate,true_positive_rate, color='yellow',label = 'Smote ratio = 0.3')


	"""
	y_score_0 = classifier.decision_function(x_val)
	y_pred_0 = classifier.predict(x_val)

	 
	#Smote ratio 0.1
	sm = SMOTE(ratio=0.1)
	x_train10, y_train10 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train10, y_train10)
	y_score_10 = classifier.decision_function(x_val)
	y_pred_10 = classifier.predict(x_val)

	#Smote ratio 0.2
	sm = SMOTE(ratio=0.2)
	x_train20, y_train20 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train20, y_train20)
	y_score_20 = classifier.decision_function(x_val)
	y_pred_20 = classifier.predict(x_val)

    #Smote ratio 0.3
	sm = SMOTE(ratio=0.3)
	x_train30, y_train30 = sm.fit_sample(x_train, y_train)
	classifier.fit(x_train20, y_train20)
	y_score_20 = classifier.decision_function(x_val)
	y_pred_20 = classifier.predict(x_val)
	"""

	"""
	precision = dict()
	recall = dict()
	average_precision = dict()
	precision, recall, thresholds = precision_recall_curve(y_val,y_score_0)
	average_precision = average_precision_score(y_val,y_score_0)
	plt.plot(recall, precision, color='red', lw=2, label='Precision-recall curve for classifier with no sampling (area = {1:0.2f})'
                   ''.format(0, average_precision))
  	"""
  	plt.plot([0, 1], [0, 1],linestyle='--')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.05])

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="lower right")
	plt.title('Receiver Operating Characteristic for %s' %label)
	plt.show()


def classication_results(y_test,y_predict,classifier_name):
	TP, FP, FN, TN = 0, 0, 0, 0
	for i in range(len(y_predict)):
		if y_test[i]==1 and y_predict[i]==1:
			TP += 1
		if y_test[i]==0 and y_predict[i]==1:
			FP += 1
		if y_test[i]==1 and y_predict[i]==0:
			FN += 1
		if y_test[i]==0 and y_predict[i]==0:
			TN += 1
	print ('The confusion matrix for ' +classifier_name )
	print ('TP: '+ str(TP))
	print ('FP: '+ str(FP))
	print ('FN: '+ str(FN))
	print ('TN: '+ str(TN))

def smote_testing_for_class(classifier, label):
	x_train, x_val, y_train, y_val = prepare_and_split_data()

	
	classifier.fit(x_train, y_train)
	y_score_no_sampling = classifier.decision_function(x_val)
	y_pred_no_sampling = classifier.predict(x_val)
	
	#classication_results(y_val,y_pred_no_sampling, label+' with no sampling')
	precision = dict()
	recall = dict()
	average_precision = dict()
	precision, recall, thresholds = precision_recall_curve(y_val,y_score_no_sampling)
	average_precision = average_precision_score(y_val,y_score_no_sampling)

	plt.plot(recall, precision, color='red', lw=2,label='Precision-recall curve for classifier with no sampling (area = {1:0.2f})'
                   ''.format(0, average_precision))
	
	plt.plot([0, 1], [0, 1],linestyle='--')
	#plt.xlim([0.0, 1.0])
	#plt.ylim([0.0, 1.05])

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc="lower right")
	plt.show()

