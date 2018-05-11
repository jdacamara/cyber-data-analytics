import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE


data = pd.read_csv('data_for_student_case.csv')
data = data[data.simple_journal != 'Refused']

#print data.simple_journal.value_counts()
#print data.iloc[0]

data['simple_journal'].replace(['Chargeback'], 1, inplace=True)
data['simple_journal'].replace(['Settled'], 0, inplace=True)

model_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode','simple_journal', 'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode' ]

relevent_data = data[model_variables]



relevant_data_enconded = pd.get_dummies(relevent_data)

#print relevant_data_enconded


training_features, test_features, \
training_target, test_target, = train_test_split(relevant_data_enconded.drop(['simple_journal'], axis=1),
                                               relevant_data_enconded['simple_journal'],
                                               test_size = .1,
                                               random_state=12)


x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                  test_size = .1,
                                                  random_state=12)

sm = SMOTE(random_state=12, ratio = 0.1)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)


clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
clf_rf.fit(x_train_res, y_train_res)


print 'Validation Results'
print clf_rf.score(x_val, y_val)
print recall_score(y_val, clf_rf.predict(x_val))
print '\nTest Results'
print clf_rf.score(test_features, test_target)
print recall_score(test_target, clf_rf.predict(test_features))


