import pandas as pd

from matplotlib import pyplot
from data import TEST_DATA, TRAINING_DATA_1
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

#Plot the auto-correlation and partial autocorrelation for L_T1 sensor
TRAINING_DATA_1["DATETIME"] = pd.to_datetime(TRAINING_DATA_1["DATETIME"])

TRAINING_DATA_1.index = TRAINING_DATA_1["DATETIME"]
del TRAINING_DATA_1["DATETIME"]

lt_1 = pd.DataFrame(TRAINING_DATA_1["L_T1"])

resampled_lt1 = lt_1.resample("H").mean()

#Choose a lags of 40, otherwise takes to long
plot_pacf(resampled_lt1, lags = 40)
plot_acf(resampled_lt1, lags = 40)

pyplot.show()