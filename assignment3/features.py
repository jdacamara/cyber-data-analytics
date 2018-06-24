import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parse_data import covert_to_dataframe

def clean_data_set(dataset):
	clean_dataset = dataset.loc[dataset['Label'] != "Background\n"]

	clean_dataset['StartTime'] = pd.to_datetime(clean_dataset['StartTime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
	clean_dataset['Packet'] = clean_dataset['Packet'].fillna(0)
	clean_dataset['Bytes'] = clean_dataset['Bytes'].fillna(0)
	clean_dataset['Duration'] = clean_dataset['Duration'].fillna(0)

	clean_dataset['Packet'] = clean_dataset['Packet'].astype(int)
	clean_dataset['Bytes'] = clean_dataset['Bytes'].astype(int)

	clean_dataset['Label'] = clean_dataset['Label'].map({"LEGITIMATE\n": 1, "Botnet\n": 0})

	return clean_dataset

def host_data(dataset, ip):
	host_data = dataset.loc[(dataset['Source'] == ip) | (dataset['Destination'] == ip)]
	host_data = host_data.reset_index()

	return host_data





if __name__ == "__main__":
	dataset = covert_to_dataframe("capture20110818.pcap.netflow.labeled")
	clean_dataset = clean_data_set(dataset)

	infected_host = "147.32.84.208"

	infected_host_data = host_data(clean_dataset, infected_host)


	sns.countplot(x="Protocol", data = infected_host_data)
	plt.show()
	plt.close()

	normal_dataset=clean_dataset.loc[clean_dataset['Label'] == 1.0]
	sns.countplot(x="Protocol", data = normal_dataset)
	plt.show()
	plt.close()


	packets = []
	bytes = []

	labels = ( infected_host, 'Legitimate', 'Botnet')

	packets.append(infected_host_data.Packet.mean())
	bytes.append(infected_host_data.Bytes.mean())

	normal_dataset = clean_dataset.loc[clean_dataset['Label'] == 1.0]

	packets.append(normal_dataset.Packet.mean())
	bytes.append(normal_dataset.Bytes.mean())

	botnet_dataset = clean_dataset.loc[clean_dataset['Label'] == 0.0]

	packets.append(botnet_dataset.Packet.mean())
	bytes.append(botnet_dataset.Bytes.mean())

	packet_bar_chart = {'Label' : labels, 'Packets': packets}
	bytes_bar_chart = {'Label' : labels, 'Bytes': bytes}

	packets_frame = pd.DataFrame(packet_bar_chart)
	bytes_frame = pd.DataFrame(bytes_bar_chart)

	sns.barplot(x = 'Label', y = 'Packets', data = packets_frame)
	plt.show()
	plt.close()

	sns.barplot(x = 'Label', y = 'Bytes', data = bytes_frame)
	plt.show()
	plt.close()


#infected_discritized = pd.DataFrame()

#infected_discritized['Packet'] = dat




'''
sns.countplot(data=infected_host_data)
plt.figure()
plt.show()
'''
#clean_dataset['StartTime'] = pd.to_datetime(clean_dataset['StartTime'], format='%Y-%m-%d %H:%M:%S')
#clean_dataset = clean_dataset.loc[clean_dataset['StartTime']]


#print( clean_dataset)


