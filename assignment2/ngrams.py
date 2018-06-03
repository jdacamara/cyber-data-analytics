from pandas import datetime, read_csv, qcut
from matplotlib import pyplot
from data import TEST_DATA, TRAINING_DATA_1
import numpy as np
import nltk
from nltk.util import ngrams as NG

from sax import SAX

def discretize(dataset, filter = 'L_T1',readings_per_letter = 1 ,alphabet_size = 3 , sliding = 0, plot = 0):
	column = dataset[filter]
	s = SAX(len(column)/readings_per_letter, alphabet_size)

	# get the letter representation of the data
	if sliding:
		(xString, xIndices) = s.sliding_window(column, sliding, 1) # get letter rep for sliding windows
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
	ngrams = NG(letter_rep, n)
	fdist = nltk.FreqDist(ngrams)
	# find n
	sum = 0
	for k, v in fdist.items():
		# plus-one smoothing for the ngrams
		fdist.__setitem__(k, v+1)
		sum = sum + v+1
	# normalize
	for k, v in fdist.items():
		fdist.__setitem__(k, v/sum)

	subgrams = NG(letter_rep, n - 1)
	prior_fdist = nltk.FreqDist(subgrams)
	# find n
	sum = 0
	for k, v in prior_fdist.items():
		prior_fdist.__setitem__(k, v + 1)
		sum = sum + v+1
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


# Discretize the data and train an N-grams model
n = 5
fdist, f2dist = ngram(discretize(TRAINING_DATA_1), n)

# convert the testset to SAX
discrete_test_data = discretize(TEST_DATA)

# loop through the test data
conditional_prob = []
print(discrete_test_data)
for i in range(n, len(discrete_test_data)):
	post = discrete_test_data.__getitem__(i)
	prior = tuple(list(discrete_test_data[i-(n-1):i]))

	print (i)

	# analyze the observed values in the test set
	# If P(n_t | n_t-2, n_t-1) < a, raise alarm
	a = 0.1
	conditional_prob.append(get_cond_prob(fdist, f2dist, post, prior))

# plot
pyplot.plot(conditional_prob[1:100])
pyplot.axhline(y=a, color='r', linestyle='-')
pyplot.show()