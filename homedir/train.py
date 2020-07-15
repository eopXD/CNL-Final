from joblib import dump, load
import numpy as np
import os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def lookup(key, database):
    equal = (database == key).all(axis=-1)

    if np.max(equal) == 0:
        return -1

    return np.argmax(equal)

if __name__ == "__main__":
    num_ap = 6
    
    for ap in range(num_ap):
        os.system("./tail_packet 1 packet/packet"+str(ap+1)+" > test"+str(ap+1)+".csv")

    raw_data_list = [np.loadtxt("labeled_packets/labeled_packet"+str(ap+1)+".csv", delimiter=',', dtype='U') for ap in range(num_ap)]

    unique_entry = np.unique(raw_data_list[0][:,2:], axis=0)

    for ap in range(1, 6):
        unique_entry = np.array([entry for entry in np.unique(raw_data_list[ap][:,2:], axis=0) if lookup(entry, unique_entry) != -1])

    print("finish unique")

    feature = []
    x_label = []
    y_label = []
    mac = []

    for entry in unique_entry:
        rss_avg = []

        for ap in range(num_ap):
            select = (raw_data_list[ap][:,2:] == entry).all(axis=-1)
            rss = raw_data_list[ap][select,1].astype(float)
            rss = rss[rss > -100]

            if rss.size == 0:
                continue

            rss_avg.append(np.average(rss))

        if len(rss_avg) != num_ap:
            continue

        x_label.append(int(entry[0]))
        y_label.append(int(entry[1]))
        mac.append(entry[2])
        feature.append(rss_avg)

    print("finish process")

    scale = StandardScaler()
    scale.fit(feature)
    dump(scale, 'x_scale')
    dump(scale, 'y_scale')

    x_model = MLPRegressor(hidden_layer_sizes=(100, 100), batch_size=4, max_iter=2000, alpha=0.002)
    x_model.fit(scale.transform(feature), x_label)
    dump(x_model, 'x_model')

    y_model = MLPRegressor(hidden_layer_sizes=(100, 100), batch_size=4, max_iter=2000, alpha=0.002)
    y_model.fit(scale.transform(feature), y_label)
    dump(y_model, 'y_model')
