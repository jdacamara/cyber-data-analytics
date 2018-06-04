from pandas import datetime, read_csv

def parser(x):
	return datetime.strptime(x, '%d/%m/%y %H')

def get_headers():
	return ['DATETIME','L_T1','L_T2','L_T3','L_T4','L_T5','L_T6','L_T7','F_PU1','S_PU1','F_PU2','S_PU2','F_PU3','S_PU3','F_PU4','S_PU4','F_PU5','S_PU5','F_PU6','S_PU6','F_PU7','S_PU7','F_PU8','S_PU8','F_PU9','S_PU9','F_PU10','S_PU10','F_PU11','S_PU11','F_V2','S_V2','P_J280','P_J269','P_J300','P_J256','P_J289','P_J415','P_J302','P_J306','P_J307','P_J317','P_J14','P_J422','ATT_FLAG']

def get_data(filename):
	series = read_csv(filename, header=0, parse_dates=[0], date_parser=parser)
	return series

def make_array(data):
	return [x for x in train]

def get_cyclic_headers():
	return ['L_T1','L_T2','L_T3','L_T4','L_T5','L_T6','L_T7']

TEST_DATA = get_data("BATADAL_test_dataset.csv")
TRAINING_DATA_1 = get_data("BATADAL_dataset03.csv")
TRAINING_DATA_2 = get_data("BATADAL_dataset04.csv")
