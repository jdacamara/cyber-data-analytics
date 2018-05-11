from imblearn.over_sampling import SMOTE as SSMOTE


class SMOTE(object):
    X_res = None
    y_res = None

    def __init__(self, data_set, ratio):
        data, data_labels = data_set
        sm = SSMOTE(ratio=ratio)
        self.X_res, self.y_res = sm.fit_sample(data, data_labels)

    def get_X(self):
        return self.X_res

    def get_y(self):
        return self.y_res

    def get_smote_data(self):
        return self.X_res, self.y_res