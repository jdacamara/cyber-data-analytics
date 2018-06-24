import random
import operator
import time

from parse_data import Packet, read_file

class MinWise:

	def __init__(self, k = 10):
		self.min_wise_list = []
		self.k = k 
		self.max_rand = 0 

	def update_min_wise_list(self, rand, value):
		if len(self.min_wise_list) == 0 :
			self.min_wise_list.append((rand,value))
			self.max_rand = rand
			return

		if len(self.min_wise_list) < self.k or self.max_rand > rand:
			index = 0 

			for r, v in self.min_wise_list:
				if r > rand:
					self.min_wise_list.insert(index, (rand,value))
					self.min_wise_list = self.min_wise_list[:self.k]
					self.max_rand = self.min_wise_list[-1][0]
					return

				index = index + 1 


def get_top_k(list_values, k = 10):
	top_k = {}
	for _, v in list_values:
		if v not in top_k.keys():
			top_k[v] = 1
		else:
			top_k[v] = top_k[v] + 1

	return sorted(top_k.items(), key=lambda x: x[1], reverse = True)[:k]

k_values = [25, 50, 100, 1000]
#k_values = [25]

for k_value in k_values:
	print(k_value)
	start = time.time()

	min_wise = MinWise(k = k_value)
	all_records = []
	amount_of_ip = 0 

	lines = read_file("capture20110811.pcap.netflow.labeled")

	#print("Len of lines = %s" %len(lines))


	for l in lines[1:]:

		p = Packet(l)

		
		if p.source_ip != "147.32.84.165":
			continue
	 	

		ip = p.destination_ip

		amount_of_ip = amount_of_ip + 1	
		rand = random.random()
		all_records.append((rand, ip))
		min_wise.update_min_wise_list(rand, ip)

	top_ten = get_top_k(all_records)
	#top_ten = get_top_k(min_wise.min_wise_list)

	print('Amount of botnet = %s' %amount_of_ip)

	for ip, count in top_ten:
		percent = (count / min_wise.k) * 100
		#percent = (count/amount_of_ip) * 100
		print("%s & %.1f & %s \\\\" %(ip, percent, int((percent /100) * amount_of_ip)))
		#print("%s & %.1f & %s \\\\" %(ip, percent, count))

		#print("%s & %.1f & %s \\\\" %(ip, percent, count))


	print ('It took ', time.time() - start, ' seconds.')
	print ("This was for k = %s" %min_wise.k)
