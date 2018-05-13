import pandas as pd
import warnings

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, VotingClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def create_voting_classifier():
	#Classifies that are gonna be used
	logistic_regrssion = LogisticRegression(random_state=1)
	adaBoost = AdaBoostClassifier(n_estimators=40)
	random_forest = RandomForestClassifier(n_estimators=200)
	gnb = GaussianNB()
	mlp = MLPClassifier(alpha=1)

	#VotingClassifier
	blackBox = VotingClassifier(estimators=[('lr', logistic_regrssion), ('ada', adaBoost),('rnd', random_forest), ('gnb', gnb), ('nn',mlp), ], voting='soft')
	return blackBox

def prepare_data():
	data = pd.read_csv(filename)
	data = data[data.simple_journal != 'Refused']
	
	#Changes the simple_journal value to binary
	data['simple_journal'].replace(['Chargeback'], 1, inplace=True)
	data['simple_journal'].replace(['Settled'], 0, inplace=True)

	#Sort the important columns
	model_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode','simple_journal', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode' ]
	relevent_data = data[model_variables]

	#Change the value to one hot-encoding
	relevant_data_enconded = pd.get_dummies(relevent_data)
	return relevant_data_enconded


def black_box(filename = '../data_for_student_case.csv'):

	relevant_data_enconded = prepare_data()

	#split the data
	training_features, test_features, \
	training_target, test_target, = train_test_split(relevant_data_enconded.drop(['simple_journal'], axis=1),
			                                           relevant_data_enconded['simple_journal'],
			                                           test_size = .1)


	x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
	                                                  test_size = .1)

	#Create and train
	blackBox = create_voting_classifier()
	blackBox = blackBox.fit(x_train, y_train)

	print 'Validation Results'
	print "The score for validation  = %s" %blackBox.score(x_val, y_val)
	print "The recall score for validation = %s " %recall_score(y_val, blackBox.predict(x_val))
	print '\nTest Results'
	print "The score for the actual test = %s " %blackBox.score(test_features, test_target)
	print "The recall score for the test = %s" %recall_score(test_target, blackBox.predict(test_features))


black_box()



