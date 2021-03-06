from pandas import datetime, DataFrame, to_datetime

def read_file(filename):
	with open(filename) as f:
		lines = f.readlines()

	return lines

def ip_and_port(ip_port):
	temp = ip_port.split(":")
	if len(temp) == 1:
		 ip = temp[0]
		 return ip
	elif len(temp) == 2:
		ip = temp[0]
		port = temp[1]
		return ip
	elif len(temp) == 6 :
		ip = ":".join(temp[:6])
		return ip
	elif len(temp) == 8 or len(temp) == 7:
		#print(temp)
		ip = ":".join(temp[:6])
		return ip
	else:
		ip = ":".join(temp[:6])
		return ip

def parse_date(date):
	date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')
	return date
		

class Packet():

	def __init__(self, line):
		self.parse_line(line)

	def parse_line(self,line):
		data = line.split()
		#print(len(data))
		
		self.date = parse_date(data[0] + " " + data[1])
		self.duration = float(data[2])

		if data[3] == "TCP":
			self.protocol = 1
		elif data[3] == "UDP":
			self.protocol = 2
		elif data[3] != 'ICMP':
			self.protocol = 3
		else:
			self.protocol = None
			

		self.source_ip = ip_and_port(data[4])
		self.destination_ip = ip_and_port(data[6])
		self.flags = data[7]
		self.tos = int(data[8])
		self.packets = int(data[9])
		self.bytes = int(data[10])
		self.flows = int(data[11])

		if data[12] == "Botnet":
			self.label = 1
		elif data[12] == "Background":
			self.label = 0
		elif data[12] != 'LEGITIMATE':
			self.label = 2
		else:
			self.label = None
			#print (data[12])

	def convert_to_list(self, position):
		result = []

		if position % 1000000 == 0: 
			print(position)
		#result.append(position)
		result.append(self.date)
		result.append(self.duration)
		result.append(self.protocol)
		result.append(self.source_ip)
		result.append(self.destination_ip)
		result.append(self.flags)
		result.append(self.tos)
		result.append(self.packets)
		result.append(self.bytes)
		result.append(self.flows)
		result.append(self.label)

		return result 



#LINES = read_file("capture20110811.pcap.netflow.labeled")

'''
packets = []
for l in lines[1:]:
	packets.append(Packet(l))

#print((len(lines) -1) == len(packets))
'''

def covert_to_dataframe(filename):
	columns=['StartTime','Duration', 'Protocol', 'Source','Direction','Destination',
	 'Flag','Tos','Packet','Bytes','Flows','Label']

	list_of_packets = []


	with open(filename) as netflow_file:
		for i, line in enumerate(netflow_file):
			
			
			flow = []
			if i > 0:

				#packet = Packet(line)
				#list_of_packets.append(packet.convert_to_list(i))


				
				data = line.split("\t")

				if len(data) >= 13:
					for attribute in data:
						attribute.strip()	
						if len(attribute) == 0:
							flow = data.remove(attribute)

				if len(data) >= 3:
					data[3] = data[3].split(':')[0]
				if len(data) >= 5:
					data[5] = data[5].split(':')[0]
				

				if flow:
					list_of_packets.append(flow)
				else:
					list_of_packets.append(data)



	dataframe = DataFrame(list_of_packets, columns = columns)
	#dataframe.to_csv("cvs_data", sep=',')

	return dataframe