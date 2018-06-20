from countminsketch import CountMinSketch
from pandas import datetime, read_csv



def make_array(data):
	return [x for x in train]

#Get the data from the dataset and can be called up by other scripts.
TRAINING_DATA_1 = get_data("capture20110811.pcap.netflow.labeled")

sketchW = CountMinSketch(1000, 10)  # table size=1000, hash functions=10
#simulate a stream of readings
for i in range(1,len(TRAINING_DATA_1)):
    sketchW.add(TRAINING_DATA_1[i])
