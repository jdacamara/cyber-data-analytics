import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#from assignment1 import get_date_from_transaction

converter = {
	'MXN' : 0.051,
	'GBP' : 1.35,
	'AUD' : 0.75,
	'NZD' : 0.70,
	'SEK' :	0.12,
}


def get_data(file_name = 'data_for_student_case.csv'):
	data = open(file_name,'r')

	(issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
	verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in range(10)]
	(issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
	verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in range(10)]

	data.readline()
	fraud_data = []
	benign_data = []
	refused_data = []


	for line in data:
		stripped = line.strip().split(',')
		label = stripped[9]

		if label == 'Refused':
			refused_data.append(stripped)
		elif label == 'Chargeback':
			fraud_data.append(stripped)
		else:
			benign_data.append(stripped)

	return fraud_data, benign_data, refused_data


def heatmap(x_value, y_values, z_values, title):
	plt.clf()
	plt.title(title)
	plt.ylabel('y')
	plt.xlabel('x')

def get_amount(arr):
	return int(arr[5])

def get_currency_code(arr):
	return arr(6)

def get_all_values_from_column(data, key, value):
	dic = {}

	for arr in data:
		label = arr[key]
		if label in dic.keys():
			dic[label].append(arr[value])
		else:
			dic[label] = [arr[value]]

	return dic


def amount_currency_scatterplot():
	fraud_data, benign_data, refused_data = get_data()
	fraud_dic = get_all_values_from_column(fraud_data, 6, 5)
	
	fraud_keys = fraud_dic.keys()
	counter = 1
	temp = {}
	for key in fraud_keys:
		temp[key]= counter
		counter = counter +1 

	fraud_values = fraud_dic.values()

	x = []
	y = []

	for key in fraud_keys:
		for v in fraud_dic[key]:
			x.append(int(v))
			y.append(temp[key])

	fig, ax = plt.subplots()
	ax.scatter(y, x)
	plt.show()

def heatmap():
	fraud_data, benign_data, refused_data = get_data()
	#temp = sns.load_dataset("data_for_student_case.csv")
	fraud_dic = get_all_values_from_column(fraud_data, 6, 5)

	fraud_keys = fraud_dic.keys()

	temp = repeat_arr(fraud_keys,10)
	amounts = repeat_arr(range(0,100000,10000),5,False)

	print len(temp)
	print len(amounts)

	counted = get_amount_appeared(fraud_dic)

	encountered = []
	for key in fraud_keys:
		for amount in range(0,100000,10000):
			encountered.append(counted[key][amount])

	print len(encountered)

	
	df = pd.DataFrame({'currency': temp, 'amounts': amounts,'encountered': encountered})
	hm = df.pivot("currency", "amounts", "encountered")
	ax = sns.heatmap(hm)
	plt.show()

	#print df

def convert_to_USD(currency, amount):
	exchange_rate = converter[currency]
	return amount * exchange_rate
	

def repeat_arr(arr,times, inverse_order = True):
	result = []
	if inverse_order == True:

		for x in arr:
			for i in range(times):
				result.append(x)
	else:
		for i in range(times):
			for x in arr:
				result.append(x)

	return result


def get_amount_appeared(data):
	keys = data.keys()
	result = {}

	for key in data.keys():
		result[key] = {}
	

	for key in data:
		for i in range(0,100000,10000):
			result[key][i] = 0
			
	max_amount = 0 
	round_max = 0 
	for key in keys:

		for amount in data[key]:

			usd_amount = convert_to_USD(key, float(amount))
			if usd_amount > max_amount:
				max_amount = usd_amount
				round_max = round(float(usd_amount) / 1000) * 1000


			rounded = round(float(usd_amount) / 10000) * 10000
			#print rounded

			result[key][rounded] = result[key][rounded] + 1 

	print "Max amount encountered = %s  and round is %s" %(max_amount, round_max)

	return  result





#amount_currency_scatterplot()
heatmap()
