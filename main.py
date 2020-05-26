import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

FILE_NAME = '../../data/avocado_LD4.csv'

def read_data():
    data = []
    with open(FILE_NAME) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            data_row = []
            for col in row:
                data_row.append(float(col))
            data.append(data_row)

    return np.array(data)

data = read_data()
X = data[:, 1:3] # select 2 attributes for k means

kmeans = KMeans(n_clusters=6, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans,s=50, cmap='viridis')
plt.show()
