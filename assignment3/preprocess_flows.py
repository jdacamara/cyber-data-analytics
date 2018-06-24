from pandas import datetime
from parse_data import *
import matplotlib.pyplot as plt

def preprocess(filename):
    flows = read_file(filename)

    #parse the lines, filter out all background packets
    arr = []
    for l in flows[1:500]:
        packet = Packet(l)
        if(packet.label != 0):
            arr.append(packet)

    print(type(arr))
    print(len(arr))

    return arr

def as_training_data(arr):
    return [[x.duration, x.protocol, x.tos, x.packets, x.bytes, x.flows, x.label] for x in arr]

def as_test_data(arr):
    return [[x.duration, x.protocol, x.tos, x.packets, x.bytes, x.flows] for x in arr]


def get_labels(arr):
    return [x.label for x in arr]

def separate_botnet(data):
    bots=[]
    legit=[]
    for x in data:
        if x.label == 'Botnet':
            bots.append(x)
        else:
            legit.append(x)

    return bots, legit

def visualize(bots, legit):
    # construct the lists for the scatterplot
    durations_bots = [a.tos for a in bots]
    durations_legit = [x.tos for x in legit]
    bytes_bots = [x.protocol for x in bots]
    bytes_legit = [x.protocol for x in legit]

    #Create the plot
    plt.scatter(durations_bots, bytes_bots,c='r')
    plt.scatter(durations_legit, bytes_legit,c='g')
    plt.show()

# Pick an infected host
# filter on source ip of the infected host

# Visualize the relevant features


# Discretize those features

