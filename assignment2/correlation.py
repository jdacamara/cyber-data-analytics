import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parser(x):
	return pd.datetime.strptime(x, '%d/%m/%y %H')

series = pd.read_csv('BATADAL_dataset03.csv', header=0, parse_dates=[0], date_parser=parser)
correlated_values = series.corr()

mask = np.zeros_like(correlated_values, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(correlated_values, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()