from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from pr_curve import smote_testing_for_classifier_PR_curve, smote_testing_for_class
from sklearn import tree

classifier1 = AdaBoostClassifier(n_estimators=60)
classifier2 = linear_model.LogisticRegression(C=1e5)
classifier3 = RandomForestClassifier(n_estimators=200)
classifier4 = tree.DecisionTreeClassifier()


smote_testing_for_classifier_PR_curve(classifier1,"DecisionTree")
