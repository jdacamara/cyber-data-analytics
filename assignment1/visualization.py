import numpy as np
import datetime
import time

from itertools import groupby
from xml.dom import minidom
#from assignment1 import get_date_from_transaction


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
		if line.strip().split(',')[9] == 'Refused':
			refused_data.append(line)
		elif line.strip().split(',')[9] == 'Chargeback':
			fraud_data.append(line)
		else:
			benign_data.append(line)

	return fraud_data, benign_data, refused_data
