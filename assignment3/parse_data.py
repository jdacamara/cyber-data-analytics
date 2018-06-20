from pandas import datetime

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
		
		self.date = parse_date(data[0] + " " + data[1])
		self.duration = data[2]
		self.protocol = data[3]
		self.skipped =False
		self.source_ip =ip_and_port(data[4])
		self.destination = ip_and_port(data[6])
		self.flags = data[7]
		self.tos = data[8]
		#print (self.tos)



lines = read_file("capture20110811.pcap.netflow.labeled")

packets = []
for l in lines[1:]:
	packets.append(Packet(l))

print((len(lines) -1) == len(packets))
