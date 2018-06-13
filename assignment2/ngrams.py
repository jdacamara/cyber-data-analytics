from pandas import datetime, read_csv, qcut
from matplotlib import pyplot
from data import TEST_DATA, TRAINING_DATA_1, TRAINING_DATA_2
import numpy as np
import nltk
from nltk.util import ngrams as NG

from sax import SAX

def discretize(dataset, filter = 'L_T1',readings_per_letter = 1 ,alphabet_size = 3 , sliding = 0, move = 1, plot = 0):
	column = dataset[filter]
	s = SAX(len(column)/readings_per_letter, alphabet_size)

	# get the letter representation of the data
	if sliding:
		(xString, xIndices) = s.sliding_window(column, sliding, move) # get letter rep for sliding windows
	else:
		(xString, xIndices) = s.to_letter_rep(column[1:(1 + len(column))])

	# construct a column with letter representations
	sax = []
	for i in range(0, len(xString)):
		for x in range(0, readings_per_letter):
			sax.insert((i*x)+1, int(xString[i]))

	if plot:
		# plot the original and the letter represented values
		fig, ax1 = pyplot.subplots()
		ax1.plot(column[1:(1+len(column))])
		ax2 = ax1.twinx()
		ax2.plot(sax, 'r.')
		pyplot.show()

	return xString

# Create ngrams from a string, return ngrams and probabilities
def ngram(letter_rep, n):
	if type(letter_rep) is str:
		letter_rep = [letter_rep,]

	# print(letter_rep)

	# Aggregate all sliding windows into one frequency distribution
	fdist = nltk.FreqDist()
	prior_fdist = nltk.FreqDist()
	print(letter_rep.__getitem__(0))
	for window in range(0, len(letter_rep)):
		print(window)
		ngrams = NG(letter_rep.__getitem__(window), n)
		FD = nltk.FreqDist(ngrams)
		for k, v in FD.items():
			# print(k, v)
			fdist.__setitem__(k, fdist.__getitem__(k) + v) #Add all counts to the final freq distribution

		subgrams = NG(letter_rep.__getitem__(window), n - 1)
		FD2 = nltk.FreqDist(subgrams)
		for k, v in FD2.items():
			prior_fdist.__setitem__(k, prior_fdist.__getitem__(k) + v) #Add all counts to the final freq distribution

# Change to probabilities
	# find n_fdist
	sum = 0
	for k, v in fdist.items():
		print(k, v)
		# plus-one smoothing for the ngrams
		fdist.__setitem__(k, v + 1)
		sum = sum + v + 1
	# normalize
	for k, v in fdist.items():
		fdist.__setitem__(k, v / sum)

	# find n_prior_fdist
	sum = 0
	for k, v in prior_fdist.items():
		prior_fdist.__setitem__(k, v + 1)
		sum = sum + v + 1
	# normalize
	for k, v in prior_fdist.items():
		prior_fdist.__setitem__(k, v / sum)


	return fdist, prior_fdist

# Get the conditional probability for a value
def get_cond_prob(fdist, prior_fdist, post, prior):
	if fdist.__contains__(prior + (post,)):
		return fdist.get(prior + (post,)) / prior_fdist.get(prior)
	else:
		return 0


def _main_():
	# Discretize the data and train an N-grams model
	w = 1 # The number of readings per letter in the representation
	n = 2 # n in the n-gram
	col = 'L_T1' # the selected column
	pdist, p2dist = ngram(discretize(TRAINING_DATA_1, col, readings_per_letter=w, alphabet_size=4), n)

	for k, v in pdist.items():
		print(k, v)
	for k, v in p2dist.items():
		print(k, v)
	# Validate with training set 2
	# convert the training set 2 to SAX
	discrete_test_data = discretize(TRAINING_DATA_2, col)

	# Test with test set
	# convert the testset to SAX
	# discrete_test_data = discretize(TEST_DATA, col)

	# loop through the test data
	conditional_prob = []
	anomaly = []
	# print(discrete_test_data)
	for i in range(n, len(discrete_test_data)):
		post = discrete_test_data.__getitem__(i)
		prior = tuple(list(discrete_test_data[i-(n-1):i]))

		# print (len(discrete_test_data))

		# analyze the observed values in the test set
		# If P(n_t | n_t-2, n_t-1) < a, raise alarm
		a = 0.05
		p = get_cond_prob(pdist, p2dist, post, prior)

		if p < a:
			anomaly.append(1)
		else:
			anomaly.append(0)
		conditional_prob.append(p)


	# plot
	pyplot.plot(conditional_prob)
	# pyplot.plot(anomaly)
	pyplot.axhline(y=a, color='r', linestyle='-')
	pyplot.show()