
class Classifier(object):

	def train(self, training_set):
		print 'Training %s' % self.__class__.__name__
		train_data, train_labes = training_set
		self.clf.fit(train_data, train_labes)
			

	def test(self, test_set):
		print 'Testing %s' % self.__class__.__name__
		
		test_data, test_labels = test_set
		predicted_labels = self.clf.predict(test_data)
		
		self.report(predicted_labels, test_labels)
		return predicted_labels

class SVM(Classifier):

	def __init__(self):
		self.clf = SVC(C=10)  # gamma=0.1


class RandomForrest(Classifier):

	def __init__(self, n_features):
		self.clf = RFC(n_estimators=600, max_features=int(sqrt(n_features)))


class LogisticRegression(Classifier):

	def __init__(self):
		self.clf = LRC()