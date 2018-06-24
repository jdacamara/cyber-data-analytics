import numpy as np
import time
import math

from random import randint
from parse_data import Packet, read_file

class CountMin:

	def __init__(self, w, d, max_seed = 1024):
		self.w = w 
		self.d = d
		self.matrix = np.zeros((d, w))
		self.get_seeds(d, max_seed)
		self.count = 0		


	def get_seeds(self, d, max_seed):
		seeds  = []
		while len(seeds) < self.d:
			r = randint(0, max_seed)
			if r not in seeds:
				seeds.append(r)

		self.seeds = seeds

	def add_item(self, item):
		self.count = self.count + 1 
		for i in range(self.d):
			seed  = self.seeds[i]
			position = hash(item + str(seed)) % self.w
			self.matrix[i][position] = self.matrix[i][position] + 1

	def get_count(self,item):
		counts = []
		for i in range(self.d):
			seed  = self.seeds[i]
			position = hash(item + str(seed))  % self.w
			value = self.matrix[i][position]
			counts.append(value)

		return min(counts)

	def get_top_k(self, list_values, k = 10):
		dic = {}

		for item in list_values:
			dic[item] = self.get_count(item)

		return sorted(dic.items(), key=lambda x: x[1], reverse = True)[:k]

def optimized_countmin( delta = 0.01, epsilon =  0.00001):
	w = int(2/epsilon)
	d = int(math.log(1/delta))

	count_min = CountMin(w, d)
	return count_min

lines = read_file("capture20110811.pcap.netflow.labeled")


epsilons =  [0.0001,0.001, 0.01, 0.1]
deltas =  [0.0001,0.001, 0.01, 0.1]


hashes = [5,10,15]
k_values = [25, 50, 100, 1000]




for epsilon in epsilons:
	for delta in deltas:
		#print(delta, epsilon)

		start = time.time()
		c = optimized_countmin(delta = delta, epsilon = epsilon)
		#c = CountMin(k_value, h)
		amount_of_ip = 0
		list_of_ip = []

		for l in lines[1:]:

			p = Packet(l)
			
			if p.source_ip != "147.32.84.165":
				continue
			

			ip = p.destination_ip

			c.add_item(ip)


			amount_of_ip = amount_of_ip +1 

			if ip not in list_of_ip:
				list_of_ip.append(ip)

		results = c.get_top_k(list_of_ip)

		print ("amount_of_ip = %s" %amount_of_ip)
		for ip, estimation in results:
			percent = (estimation / amount_of_ip) *100

			print("%s & %.1f & %s \\\\" %(ip, percent, estimation))



		print ('It took ', time.time() - start, ' seconds.')
		print ("W = %s, D = %s, epsilon = %s , delta = %s" %(c.w, c.d, epsilon, delta))