import numpy as np
import statistics as st

from datetime import  timedelta
from statsmodels.tsa.arima_model import ARMA
from pandas import datetime, read_csv, DataFrame, parser
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from data import TEST_DATA, TRAINING_DATA_1,TRAINING_DATA_2, get_cyclic_headers
from statsmodels.tsa.stattools import arma_order_select_ic
from matplotlib.dates import MonthLocator, WeekdayLocator, DateFormatter

def predict(coef, history):
	yhat = 0.0
	for i in range(1, len(coef)+1):
		yhat += coef[i-1] * history[-i]
	return yhat

def difference(dataset):
	diff = list()
	for i in range(1, len(dataset)):
		value = dataset[i] - dataset[i - 1]
		diff.append(value)
	return np.array(diff)

def aic(data, headers):
	aic_orders = {}

	for column_name in headers:
		column = data[column_name]

		order_selection = arma_order_select_ic(column.values, max_ar = 4, max_ma = 2, ic = "aic")

		aic_orders[column_name] = order_selection.aic_min_order
		#print ("Column name  = %s and order_selection = %s" %(column_name, order_selection.aic_min_order))

	return aic_orders


def arma(training_dataset, test_data, columns = ['L_T1'], add_space = True):
	aic_orders = aic(training_dataset, columns)


	predictions = {}
	bounds = {}

	attack_flags_name = "" 



	for column in columns:


		#Since the second dataset contains spaces in the headers
		if add_space:
			test_column_name = " " + column 
		else:
			test_column_name = column

		series = training_dataset[column].values
		training_data = [x for x in series]

		test_values = test_data[test_column_name].values
		history = [x for x in test_values[:5]]
		test_values = test_values[5:]

		std = st.stdev(training_data)
		mean = st.mean(training_data)
		
		predictions[column] = list()

		model = ARMA(training_data, order =(aic_orders[column]))
		model_fit = model.fit( disp=False)

		for t in range(len(test_values)):
			ar_coef, ma_coef = model_fit.arparams, model_fit.maparams
			resid = model_fit.resid
			#print (resid)
			diff = difference(history)
			yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
			predictions[column].append(yhat)
			obs = test_values[t]
			history.append(obs)
			#print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
		mse = mean_squared_error(test_values, predictions[column])
		rmse = np.sqrt(mse)
		print('Test RMSE: %.3f for column %s' %( rmse, column))	

		dates = [x for x in test_data['DATETIME']]
		dates = dates[5:]

		train_dates = [x for x in training_dataset['DATETIME']]

		#plot_arma(column, dates, predictions[column], test_values,  std, mean)
		#plot_residual(column,train_dates, model_fit.resid)
		#write_out(column, predictions[column], model_fit.resid, mse)

		upperbound = mean + (3 * std)
		lowerbound = mean - (3 * std)
		bounds[column] = (lowerbound,upperbound)

	if add_space:
		attack_flags_name = ' ATT_FLAG'
	else:
		attack_flags_name = 'ATT_FLAG'

	attack_flags =  test_data[attack_flags_name]

	overlap_results(attack_flags, predictions,bounds,columns)

def overlap_results(attack_flags,predictions, bounds, columns):
	true_positive = false_negative = number_of_attacks =  0

	for i in range(5, len(attack_flags)):
		if int(attack_flags[i]) == 1:
			number_of_attacks = number_of_attacks +1 

			registered = False
			for column in columns :
				lowerbound, upperbound = bounds[column]
				if predictions[column][i] < lowerbound or predictions[column][i] > upperbound:
					true_positive = true_positive + 1
					registered = True
					break
			if not registered:
				false_negative = false_negative +1 

	print("false_negative = %s " %false_negative)
	print("true_positive = %s" %true_positive)
	print("number_of_attacks = %s" %number_of_attacks)



def plot_residual(column_name, dates, residuals):

	residuals = DataFrame(residuals)
	#results = [x for x in residuals]
	residuals.plot()
	#pyplot.plot(results)
	#pyplot.show()
	pyplot.savefig("arma_results/residuals/error/%s" %column_name)
	pyplot.close()


	residuals.plot(kind='kde')
	pyplot.savefig("arma_results/residuals/density/%s" %column_name)
	pyplot.close()
	#pyplot.show()
	#print(residuals.describe())
	

	"""
	result = [x for x in residuals]

	pyplot(result)
	
	pyplot.ylabel(column_name)
	pyplot.savefig("arma_results/residuals/%s" %column)
	pyplot.cla()"""
	

def plot_arma(column_name, dates,y,y2, std, mean):
	day_before = dates[0] - timedelta(days=10)
	last_day = dates[-1] - timedelta(days=10)

	upperbound = mean + (3 * std)
	lowerbound = mean - (3 * std)

	pyplot.axhline(y=upperbound, color = 'red')
	pyplot.axhline(y=lowerbound, color = 'red')

	pyplot.xlim(day_before, last_day)

	#print ("Length of dates = %s" %len(dates))
	#print ("Length of predictions = %s" %len(predictions[column]))

	pyplot.plot(dates,y)
	pyplot.plot(dates,y2)
	pyplot.xlabel("Date")
	pyplot.ylabel(column_name)
	pyplot.savefig("arma_results/arma/%s" %column_name)
	pyplot.close()

def write_out(column_name, predictions, residuals, mse):
	residuals = DataFrame(residuals)
	string_predictions = list_to_string(predictions)
	string_residuals = residuals.to_string()

	f = open("arma_results/info/%s.txt"%column_name,"w+") 
	f.write(string_predictions)
	f.write("\n")
	f.write(string_residuals)
	f.write("\n")
	f.write(str(mse))
	f.close()

def list_to_string(l):
	result = ""
	for x in l:
		result = result + str(x) + "," 

		result = result[:len(result)-1]

	return result


headers = get_cyclic_headers()

arma(TRAINING_DATA_1, TRAINING_DATA_2, columns = headers, add_space = True)

#arma(TRAINING_DATA_1, TRAINING_DATA_2, columns = headers)