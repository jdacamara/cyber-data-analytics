import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from  datetime import datetime

"""
data = pd.read_csv("data_for_student_case.csv")
values = records.values
#print records.head(10)


y = data['simple_journal']
X = data.ix[:, data.columns != 'simple_journal']

plt.imshow(values,cmap = 'hot', interpolation = 'nearest')
plt.show()

#sns.heatmap(records)
data = [go.Heatmap( z=records.values.tolist(), colorscale='Viridis')]
py.iplot(data, filename='pandas-heatmap')"""

BOOKINGDATE = 1
SIMPLEJOURNAL = 9
CARDID = 16

transactions = []

training_set = None
test_set = None

#Parses string to datetime according to given format
def get_date_from_transaction(tx, date_feature):
	return datetime.strptime(tx[date_feature], '%Y-%m-%d %H:%M:%S')


#Read the transactions from the csv file
with open('data_for_student_case.csv') as f:
    print 'Reading and pre-processing data...'

    for i, line in enumerate(f.read().split('\n')):
        if i == 0:
            continue

        tx = line.split(',')
        if len(tx) > 1:
            tx_dict = {feature: value for feature, value in enumerate(tx)}
transactions.append(tx_dict)


#Sort the transactions 
print 'Sorting transactions on date...'
transactions = sorted(transactions, key=lambda tx: get_date_from_transaction(tx, BOOKINGDATE))

#Get all fraudulent transactions and get a set of all fraudulent cards
fraud_transactions = []
fraudulent_cards = set()
for tx in transactions:
    if tx[SIMPLEJOURNAL] == 'Chargeback':
        fraud_transactions.append(tx)
fraudulent_cards.add(tx[CARDID])
