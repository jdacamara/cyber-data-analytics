import numpy as np

from matplotlib import pyplot
from pandas import concat, DataFrame, read_csv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from data import TEST_DATA, TRAINING_DATA_1, TRAINING_DATA_2

TRAINING_DATA_1 = read_csv("BATADAL_dataset03.csv", index_col = 'DATETIME', parse_dates = True)
TRAINING_DATA_2 = read_csv("BATADAL_dataset04.csv", index_col = 'DATETIME', parse_dates = True)
TRAINING_DATA_2.rename(columns=lambda x: x.strip(), inplace = True)
TEST_DATA = read_csv("BATADAL_test_dataset.csv",  index_col = 'DATETIME', parse_dates = True)

def set_attacks(test_set):
	test_set = test_set.assign(ATT_FLAG=np.zeros(test_set.shape[:][0]))
	test_set.loc[:,'ATT_FLAG'] = test_set.loc[:, 'ATT_FLAG'].map({0: -999})
	test_set.loc['2017-01-16 09:00:00':'2017-01-19 06:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-01-30 08:00:00':'2017-02-02 00:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-02-09 03:00:00':'2017-02-10 09:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-02-12 01:00:00':'2017-02-13 07:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-02-24 05:00:00':'2017-02-28 08:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-03-10 14:00:00':'2017-03-13 21:00:00', 'ATT_FLAG'] = 1
	test_set.loc['2017-03-25 20:00:00':'2017-03-27 01:00:00', 'ATT_FLAG'] = 1

	return test_set


def fill_dataset(dataset):
	return dataset.fillna(dataset.median(axis = 0))

def transform_data(dataset):
	scalar = StandardScaler()
	x = dataset.iloc[:,0:43].values
	x = scalar.fit_transform(x)
	y = dataset.iloc[:, 43]

	return x,y

def overlap_results(actual_values, predictions):
	false_negative = true_positive = 0

	for i in range(actual_values.shape[0]):
		if(actual_values[i] == 1 and predictions[i] == 1):
			true_positive = true_positive + 1
		if(actual_values[i] == 1 and predictions[i] != 1):
			false_negative = false_negative + 1 

	#print("false_negative = %s " %false_negative)
	#print("true_positive = %s" %true_positive)

	return true_positive,false_negative


def pca_task(pca, t, index, y, plot_figure = True):
	pca_model = pca.transform(t)
	residual = t - pca.inverse_transform(pca_model)
	spe = np.square(np.linalg.norm(residual, axis = 1))
	spe = spe / max(spe)


	#square predictions error
	pyplot.plot(spe)
	pyplot.plot(y.map({-999: -0.1, 0: 0, 1: 1}).values)
	if plot_figure:
		pyplot.show()
	pyplot.close()

	threshold = 0.09
	na = spe > threshold

	pyplot.figure(figsize = [25, 10])
	pyplot.plot(na)
	pyplot.plot(y.map({-999: -0.1, 0: 0, 1: 1}).values)
	if plot_figure:
		pyplot.show()
	pyplot.close()

	return overlap_results(y.values, na)

def best_components(training_data,max_amount_of_components, t, index,y):
	results = {}
	for i in range(1, max_amount_of_components):
		pca = PCA(n_components = i)
		pca.fit(training_data)

		true_positive,false_negative = pca_task(pca , t, index, y, plot_figure = False)

		results[i] = (true_positive,false_negative)

	right = []
	left = []
	for key in results.keys():
		true_positive,false_negative = results[key]
		right.append(true_positive)
		left.append(false_negative)
		#git print ("Key = %s has tp = %s and fn = %s" %(key, true_positive, false_negative))

	width = 0.35  
	ind = np.arange(len(right))
	fig, ax = pyplot.subplots()
	ax.bar(ind,right)

	ax.bar(ind+ width, left)
	pyplot.xlabel("Number of components")
	pyplot.ylabel("Amount")
	pyplot.legend([ "True positive","False negative"])

	pyplot.show()





test_set = set_attacks(TEST_DATA)

test_set = fill_dataset(test_set)
training_set1 = fill_dataset(TRAINING_DATA_1)
training_set2 = fill_dataset(TRAINING_DATA_2)

x1,y1 = transform_data(training_set1)
x2,y2 = transform_data(training_set2)
z, yz = transform_data(test_set)

results = {}
#for i in range(1,44):
pca = PCA(n_components = 7)
pca.fit(x1)

true_positive,false_negative = pca_task(pca ,x1, training_set1.index, y1)

best_components(x1,44,x2, training_set2.index, y2)
