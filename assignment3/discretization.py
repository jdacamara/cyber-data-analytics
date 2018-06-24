import pandas as pd
import math

from parse_data import covert_to_dataframe
from features import clean_data_set, host_data

def mapping(x, split):
	for i, s in enumerate(split):
		if x <= s:
			return i
	return len(split)

def ordinal_rank(bins, column_name, dataset):
	percentile = int(100 / bins)
	split_list = []

	for p in range(percentile, 99, percentile):
		rank = math.ceil((p/100.0)*len(dataset[column_name])*1.0)
		value = sorted(dataset[column_name])[int(rank)]
		split_list.append(value)
	return split_list


def encode_netflow(netflow, features):
	code = 0
	space_size = features[0] * features[1]

	for i in range(len(features)):
		code = code + netflow[i] * space_size / features[i]
		space_size = space_size / features[i]

	return code


infected_host = "147.32.84.208"

dataset = covert_to_dataframe("capture20110818.pcap.netflow.labeled")
clean_dataset = clean_data_set(dataset)

infect_host_data = host_data(clean_dataset, infected_host)

infected_discretized = pd.DataFrame()

split_list_for_packets = ordinal_rank(4, 'Packet', clean_dataset)

infected_discretized['Packet'] = infect_host_data['Packet'].apply(lambda x : mapping(x, split_list_for_packets))
infected_discretized['Protocol'] = pd.factorize(infect_host_data['Protocol'])[0]
features = [infected_discretized[name].nunique() for name in infected_discretized.columns[0:2]]
infected_discretized['Code'] = infected_discretized.apply(lambda x : encode_netflow(x, features), axis = 1)

discretized = pd.DataFrame()

discretized['Packet'] = clean_dataset['Packet'].apply(lambda x : mapping(x, split_list_for_packets))
discretized['Protocol'] = pd.factorize(clean_dataset['Protocol'])[0]

features = [discretized[name].nunique(name) for name in discretized.columns[0:2]]


discretized['Code'] = discretized.apply(lambda x : encode_netflow(x, features), axis = 1)
clean_dataset['Code'] =  discretized['Code']

discretized['Source'] = clean_dataset['Source']
discretized['Destination'] = clean_dataset['Destination']
discretized['StartTime'] = clean_dataset['StartTime']


