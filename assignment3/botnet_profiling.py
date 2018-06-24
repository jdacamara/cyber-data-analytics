import numpy as np
from hmmlearn import hmm
np.random.seed(42)

from preprocess_flows import *

# sliding window function
def window(iterable, size=2):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win
    return win


# Filter for a certain infected host (147.32.84.191)
def split_by_ip(data, ip):
    host_data = []
    other_data = []
    for x in data:
        if x.source_ip == ip:
            host_data.append(x)
        else:
            other_data.append(x)
    return host_data, other_data

# Load scenario 10 and split the dataset into a host set and the other flows
host_data, other_data = split_by_ip(preprocess("capture20110818.pcap.netflow.labeled"), "147.32.84.191")

# Turn it into sequential data (slide window)
window_size = 4
host_data_seq = window(as_training_data(host_data), window_size)
other_data_seq = window(as_training_data(other_data), window_size)

# Initiate two HMM models (one for the host data, one for the other flow data)
host_model = hmm.GaussianHMM(n_components=3, covariance_type="full")
other_model = hmm.GaussianHMM(n_components=3, covariance_type="full")

# Fit a HMM to the data
for x in host_data_seq:
    host_model.fit(x)
for x in other_data_seq:
    other_model.fit(x)

print(host_model.get_params())
print(other_model.get_params())

# Expected and observed occurences