from pandas import datetime

def read_file(filename):
	with open(filename) as f:
		lines = f.readlines()

	return lines

def ip_and_port(ip_port):
	temp = ip_port.split(":")
	if len(temp) == 1:
		 ip = temp[0]
		 return ip, False
	elif len(temp) == 2:
		ip = temp[0]
		port = temp[1]
		return ip, False
	elif len(temp) == 6 :
		ip = ":".join(temp[:6])
		return ip, False
	else:
		return "", True

		#print( temp)


lines = read_file("capture20110811.pcap.netflow.labeled")


class Line():

	def __init__(self, line):
		self.parse_line(line)

	def parse_line(self,line):
		data = line.split()
		
		self.parse_date(data[0] + " " + data[1])
		self.duration = data[2]
		self.protocol = data[3]
		self.skipped =False
		ip, skipped =ip_and_port(data[4])

		if skipped:
			return

		self.source_ip = ip

		ip, skipped= ip_and_port(data[6])

		if skipped:
			return

		self.destination = ip

		self.flags = data[7]
		self.tos = data[8]
		print (self.tos)


	def parse_date(self, date):
		self.date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f')

list_of_data = []
for l in lines[1:]:
	Line(l)
