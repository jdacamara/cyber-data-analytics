from statsmodels.tsa.arima_model import ARMA
from pandas import datetime, read_csv
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from data import TEST_DATA, TRAINING_DATA_1


def arma(p,q, training_data, test_data, colum = 'L_T1'):
	series = read_csv(filename, header=0, parse_dates=[0], date_parser=parser)
	#series = series.drop(['S_PU1','F_PU2','S_PU2','F_PU3','S_PU3','F_PU4','S_PU4','F_PU5','S_PU5','F_PU6','S_PU6','F_PU7','S_PU7','F_PU8','S_PU8','F_PU9','S_PU9','F_PU10','S_PU10','F_PU11'], axis=1)
	series = series.drop(columns=['DATETIME','L_T2','L_T3','L_T4','L_T5','L_T6','L_T7','F_PU1','S_PU1','F_PU2','S_PU2','F_PU3','S_PU3','F_PU4','S_PU4','F_PU5','S_PU5','F_PU6','S_PU6','F_PU7','S_PU7','F_PU8','S_PU8','F_PU9','S_PU9','F_PU10','S_PU10','F_PU11','S_PU11','F_V2','S_V2','P_J280','P_J269','P_J300','P_J256','P_J289','P_J415','P_J302','P_J306','P_J307','P_J317','P_J14','P_J422','ATT_FLAG'])
	X = series.values
	size = int(len(X) * 0.66)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARMA(history, order=(p,q))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)
	# plot
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.show()

print (type(TRAINING_DATA_1))
colum = TRAINING_DATA_1['L_T1']
#arma(5,1, TRAINING_DATA_1, test_data)