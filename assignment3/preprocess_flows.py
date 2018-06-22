from pandas import datetime
from parse_data import *
import matplotlib.pyplot as plt

s10_flows = read_file("capture20110811.pcap.netflow.labeled")
# s11_flows = read_file("capture20110811-2.pcap.netflow.labeled")

#parse the lines, filter out all background packets
scenario10 = []
for l in s10_flows[1:]:
    packet = Packet(l)
    if(packet.label != 'Background'):
        scenario10.append(packet)

print(type(scenario10))
print(len(scenario10))

s10bots=[]
s10legit=[]
for x in scenario10:
    if x.label == 'Botnet':
        s10bots.append(x)
    else:
        s10legit.append(x)

# construct the lists for the scatterplot
durations_bots = [x.tos for x in s10bots]
durations_legit = [x.tos for x in s10legit]
bytes_bots = [x.protocol for x in s10bots]
bytes_legit = [x.protocol for x in s10legit]

#Create the plot
plt.scatter(durations_bots, bytes_bots,c='r')
plt.scatter(durations_legit, bytes_legit,c='g')
plt.show()

# Pick an infected host
# filter on source ip of the infected host

# Visualize the relevant features


# Discretize those features

