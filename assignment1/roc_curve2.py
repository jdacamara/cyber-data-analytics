from time import time

from sklearn import svm
from fraud_data import get_training_set, get_test_set

def train(self, training_set):
	print 'Training %s' % self.__class__.__name__
	start = time()
	train_data, train_labes = training_set
	self.clf.fit(train_data, train_labes)
	duration = time() - start
	print 'Training took %.2fs' % (duration)

training_set = get_test_set()
test = get_test_set()

svm = svm.LinearSVC()
svm.train(training_set)