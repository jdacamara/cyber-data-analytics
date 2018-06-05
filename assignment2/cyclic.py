from data import TRAINING_DATA_1, get_headers
from matplotlib import pyplot

#Get all headers and removes the datetime column.
columns = get_headers()
columns = columns[1:]

#Creates a plot for all column for the first 300 values
for column in columns:
	column_data = TRAINING_DATA_1[column]
	values = column_data.values
	plot_values = [x for x in values[:300]]
	pyplot.plot(plot_values)
	pyplot.ylabel(column)
	pyplot.xlabel("index")
	pyplot.savefig("cyclic/%s" %column)
	pyplot.cla()