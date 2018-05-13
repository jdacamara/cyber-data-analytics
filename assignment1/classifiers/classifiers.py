import graphviz
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

def main():
    data = pd.read_csv('../data_for_student_case.csv')
    #data = data[1:10000]
    data = data[data.simple_journal != 'Refused']

    # Changes the simple_journal value to binary
    data['simple_journal'].replace(['Chargeback'], 1, inplace=True)
    data['simple_journal'].replace(['Settled'], 0, inplace=True)

    # Sort the important columns
    train_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode',
                       'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']

    y = np.array(data['simple_journal']) # LABELS
    # Select only the training variables
    train = data[train_variables]        # TRAINING DATA
    train = pd.get_dummies(train)

    # Define classfiers
    whitebox = tree.DecisionTreeClassifier(max_depth=3)
    blackbox = svm.SVC(kernel='linear', C=1)

    # Fit & Cross-validate
    wb_scores = cross_val_score(whitebox, np.array(train), y, cv=10)  # validate white box
    print("White Box accuracy: %0.2f (+/- %0.2f)" % (wb_scores.mean(), wb_scores.std() * 2))
    # bb_scores = cross_val_score(blackbox, np.array(train), y, cv=5)  # validate black box
    # print("Black Box accuracy: %0.2f (+/- %0.2f)" % (bb_scores.mean(), bb_scores.std() * 2))

    # Split data to make an example decision tree
    x_train, x_val, y_train, y_val = train_test_split(train, y, test_size=.1)
    # Fit a decision tree to the split training data
    whitebox.fit(x_train, y_train)
    # Visualize white box rules (decision tree)
    dot_data = tree.export_graphviz(whitebox, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("Credit card fraud data - Decision tree")

    pass

if __name__ == '__main__':
    main()