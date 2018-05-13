import graphviz
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.model_selection import cross_val_score


# # Prepare and split data
# csv = pd.read_csv('..\data_for_student_case.csv')
# csv = csv[csv.simple_journal != 'Refused']
#
# # Changes the simple_journal value to binary
# csv['simple_journal'].replace(['Chargeback'], 1, inplace=True)
# csv['simple_journal'].replace(['Settled'], 0, inplace=True)
#
# # Sort the important columns
# model_variables = ['issuercountrycode', 'amount', 'currencycode', 'shoppercountrycode', 'simple_journal',
#                    'shopperinteraction', 'cardverificationcodesupplied', 'cvcresponsecode', 'accountcode']
#
# # Change the scope
# # relevant_data = csv[model_variables]
# # # Change the value to one hot-encoding
# # relevant_data_encoded = pd.get_dummies(relevant_data)
#
#
# # Load the dataset as a data frame
# # df = pd.DataFrame(relevant_data_encoded.drop(['simple_journal'], axis=1), columns=model_variables.remove('simple_journal'))
# # y = relevant_data_encoded['simple_journal'] # The labels

'''
This script is to illustrate a solid cross validation process for this competition.
We use 10 fold out-of-bag overall cross validation instead of averaging over folds.
The entire process is repeated 5 times and then averaged.

You would notice that the CV value obtained by this method would be lower than the
usual procedure of averaging over folds. It also tends to have very low deviation.

Any scikit learn model can be validated using this. Models like XGBoost and
Keras Neural Networks can also be validated using their respective scikit learn APIs.
XGBoost is illustrated here along with Ridge regression.
'''

# def R2(ypred, ytrue):
#     y_avg = np.mean(ytrue)
#     SS_tot = np.sum((ytrue - y_avg) ** 2)
#     SS_res = np.sum((ytrue - ypred) ** 2)
#     r2 = 1 - (SS_res / SS_tot)
#     return r2
#
# def cross_validate(model, x, y, folds=10, repeats=5):
#     '''
#     Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
#     model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
#     x = training data, numpy array
#     y = training labels, numpy array
#     folds = K, the number of folds to divide the data into
#     repeats = Number of times to repeat validation process for more confidence
#     '''
#     ypred = np.zeros((len(y), repeats))
#     score = np.zeros(repeats)
#     x = np.array(x)
#     for r in range(repeats):
#         i = 0
#         print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))
#         x, y = shuffle(x, y, random_state=r)  # shuffle data before each repeat
#         kf = KFold(n_splits=folds, random_state=i + 1000)  # random split, different each time
#         for train_ind, test_ind in kf.split(x):
#             print('Fold', i + 1, 'out of', folds)
#             xtrain, ytrain = x[train_ind, :], y[train_ind]
#             xtest, ytest = x[test_ind, :], y[test_ind]
#             model.fit(xtrain, ytrain)
#             ypred[test_ind, r] = model.predict(xtest)
#             i += 1
#         score[r] = R2(ypred[:, r], y)
#     print('\nOverall R2:', str(score))
#     print('Mean:', str(np.mean(score)))
#     print('Deviation:', str(np.std(score)))
#     pass


def main():
    data = pd.read_csv('../data_for_student_case.csv')
    data = data[1:10000]
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
    whitebox = tree.DecisionTreeClassifier()
    blackbox = svm.SVC(kernel='linear', C=1)

    print(blackbox)
    print(np.array(train))
    print(y)

    # Fit & Cross-validate
    # wb_scores = cross_val_score(whitebox, np.array(train), y, cv=10)  # validate white box
    # print("White Box accuracy: %0.2f (+/- %0.2f)" % (wb_scores.mean(), wb_scores.std() * 2))
    bb_scores = cross_val_score(blackbox, np.array(train), y, cv=5)  # validate black box
    print("Black Box accuracy: %0.2f (+/- %0.2f)" % (bb_scores.mean(), bb_scores.std() * 2))

    # Visualize white box rules (decision tree)
    # dot_data = tree.export_graphviz(whitebox, out_file=None,
    #                                 feature_names=train_variables,
    #                                 class_names=['Chargeback','Settled'],
    #                                 filled=True, rounded=True,
    #                                 special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph

    pass

if __name__ == '__main__':
    main()