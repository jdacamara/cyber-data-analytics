import numpy as np
import mmh3
import time
import math

from random import randint
from parse_data import Packet, LINES

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
			position = mmh3.hash(item, seed) % self.w
			self.matrix[i][position] = self.matrix[i][position] + 1

	def get_count(self,item):
		counts = []
		for i in range(self.d):
			seed  = self.seeds[i]
			position = mmh3.hash(item, seed) % self.w
			value = self.matrix[i][position]
			counts.append(value)

		return min(counts)

	def get_top_k(self, list_values, k = 10):
		dic = {}

		for item in list_values:
			dic[item] = self.get_count(item)

		return sorted(dic.items(), key=lambda x: x[1], reverse = True)[:k]

def optimized_countmin(self, delta = 0.01, epsilon =  0.00001):
	w = 2/epsilon
	d = math.log(1/delta)

	count_min = CountMin(w, d)
	return count_min



start = time.time()
c = CountMin(100,10)

amount_of_ip = 0
list_of_ip = []

for l in LINES[1:]:

	p = Packet(l)
	'''
	if p.source_ip != "147.32.84.165":
		continue
	'''

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
print ("This was for w = %s" %c.w)
