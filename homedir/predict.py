from joblib import dump, load
import numpy as np
import os, hashlib
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def lookup(key, database):
    equal = (database == key).all(axis=-1)

    if np.max(equal) == 0:
        return -1

    return np.argmax(equal)

if __name__ == "__main__":
    num_ap = 6

    x_scale = load('x_scale')
    y_scale = load('y_scale')

    x_model = load('x_model')
    y_model = load('y_model')

    x_avg, y_avg = {}, {}
    
    while True:
        for ap in range(num_ap):
            os.system("./tail_packet 2 packet/packet"+str(ap+1)+" > test"+str(ap+1)+".csv")

        raw_data_list = [np.loadtxt("test"+str(ap+1)+".csv", delimiter=',', dtype='U') for ap in range(num_ap)]

        #raw_data_list = [np.loadtxt("labeled_packets/labeled_packet"+str(ap+1)+".csv", delimiter=',', dtype='U') for ap in range(num_ap)]

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


        x_pred = x_model.predict(x_scale.transform(feature))
        y_pred = y_model.predict(y_scale.transform(feature))
        alpha = 0.4
        for i in range(len(mac)):
            if mac[i] in x_avg:
                x_avg[mac[i]] = x_avg[mac[i]] * alpha + x_pred[i] * (1. - alpha)
                y_avg[mac[i]] = y_avg[mac[i]] * alpha + y_pred[i] * (1. - alpha)
            else:
                x_avg[mac[i]] = x_pred[i]
                y_avg[mac[i]] = y_pred[i]

        #URL = "https://linux7.csie.org:9090/pos"

        #for i in range(len(mac)):
        #    PARAMS = dict()
        #    PARAMS['x'] = x_pred[i]
        #    PARAMS['y'] = y_pred[i]
        #    PARAMS['mac'] = mac[i]
        #    requests.get(url = URL, params = PARAMS, verify=False)

        def Color(mac):
            h = int.from_bytes(hashlib.sha256(mac.encode()).digest(), 'little')
            r = h % 128 + 64
            g = h // 128 % 128 + 64
            b = h // (128*128) % 128 + 64
            return '#%02x%02x%02x' % (r,g,b)
        with open('html/monitor/js/data.js', 'w') as f:
            f.write('var glob_data = [\n')
            f.write("""  [2.7, 2.0, "RasPi 1", "#0000ff"],
  [0.5, 13.0, "RasPi 2", "#0000ff"],
  [0.3, 19.4, "RasPi 3", "#0000ff"],
  [12.2, 0.5, "RasPi 4", "#0000ff"],
  [16.0, 12.5, "RasPi 5", "#0000ff"],
  [16.1, 19.0, "RasPi 6", "#0000ff"],
  """)
            for i in x_avg:
                if i not in mac: continue
                f.write('[%f,%f,"%s","%s"],\n' % (x_avg[i], y_avg[i], i, Color(i)))
            f.write(']\n')
