'''
#### Kmeans implementation assignment 1 for Scientific Data Management ####
### See test folder for tests
### see data folder for datasets

DATA SET SOURCES:
skin_noskin: 
description: https://archive.ics.uci.edu/ml/datasets/skin+segmentation
data: https://archive.ics.uci.edu/ml/machine-learning-databases/00372/

HTRU2: 
description: https://archive.ics.uci.edu/ml/datasets/HTRU2
data:https://archive.ics.uci.edu/ml/machine-learning-databases/00372/

'''

import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import sklearn as skl #useful for initial analysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

import math
import seaborn as sns #useful for splitting test and training data set and other machine learning methods
from matplotlib import pyplot as plt #useful for visualising data
import abc


from mpl_toolkits import mplot3d

import os

class KMeans:
    
    def __init__(self, filename, k_clusters=0):
        self.filename = filename
        self.k_clusters = k_clusters
        self.init_centroids = []
        self.optimized_clusters = {}
        self.raw_data = None
        self.point_cloud = []
        self.transformed_point_cloud = []
        self.training_set = None
        self.test_set = None
        self.init_strategy = None
        self.update_strategy = None

    #imports raw data and checks that it is a valid filetype.
    def import_data(self):
        print('importing data from {}'.format(self.filename))
        #read the filetype and run relevant case
        filename, ftype = os.path.splitext(self.filename)
        if ftype == '.csv':
            self.raw_data = pd.read_csv(self.filename, header=None)
            return
        if ftype == '.txt':
            self.raw_data = pd.read_csv(self.filename, sep="\t", header=None)
            return
        else:
            print('{} is an invalid filetype'.format(ftype))
            return

    # converts data from tsv (N columns) to nparray of points point cloud.
    def convert_data(self):
        if self.point_cloud is None or len(self.raw_data) is 0:
            raise Exception('now raw data available, nothing to convert')
        i = 0
        while i < len(self.raw_data):
            point = []
            for j in range(0, len(self.raw_data.columns)):
                point.append(self.raw_data[j][i])
            point = np.array(point)
            i+=1
            self.point_cloud.append(point)


    def transform_skin_noskin_data(self):
        if self.point_cloud is None or len(self.raw_data) is 0:
            raise Exception('now raw data available, nothing to transform')

        #Remove duplicates -> screws NMI, dont know why yet
        #self.point_cloud = np.unique(self.point_cloud, axis=0)

        # Remove 4th elem
        for i, point in enumerate(self.point_cloud):
            self.transformed_point_cloud.append(self.point_cloud[i][:3])

        #Normalize data
        self.transformed_point_cloud = preprocessing.normalize(self.transformed_point_cloud)


    def visualize_clusters_skin_noskin(self):

        #Source: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

        ax = plt.axes(projection='3d')

        # Data for three-dimensional scattered points
        zdata = []
        xdata = []
        ydata = []
        #self.point_cloud = np.arccosh(self.point_cloud)
        for i,key in enumerate(self.optimized_clusters.keys()):
            zdata.clear()
            xdata.clear()
            ydata.clear()
            for point in self.optimized_clusters[key][1]:
                zdata.append(self.point_cloud[point][2])
                ydata.append(self.point_cloud[point][1])
                xdata.append(self.point_cloud[point][0])

            if i == 0:
                ax.scatter3D(xdata, ydata, zdata, c='r', marker='1')
            if i == 1:
                ax.scatter3D(xdata, ydata, zdata, c='b', marker='2')
            if i == 2:
                ax.scatter3D(xdata, ydata, zdata, c='g', marker='.')
            print(len(self.optimized_clusters[key][1]))
        plt.show()
        pass

    def calc_nmi_skin_noskin_data(self):
        true = []
        for i, point in enumerate(self.point_cloud):
            if point[3] == 1:
                true.append(1)
            else:
                true.append(0)

        pred = []
        for cluster_no, key in enumerate(self.optimized_clusters.keys()):
            for i, point in enumerate(self.optimized_clusters[key][1]):
                if cluster_no == 1:
                    pred.append(1)
                else:
                    pred.append(0)
        return metrics.cluster.normalized_mutual_info_score(true, pred)

    #summary of data
    def dataSummary(self):
        #TODO
        if self.point_cloud is None or len(self.point_cloud) is 0:
            print('the data has not been created in dataSummary()')
            return False
        print('{}\n {}\n {}\n {}\n'.format(self.point_cloud.head(), self.point_cloud.info(), self.point_cloud.describe(), self.point_cloud.columns))
        if self.k_clusters == 0:
            #TODO
            pass

    #If the number of clusters needs to be updated.
    def changeClusters(self, kClusters):
        if kClusters <= 0 or isinstance(kClusters, int) is False:
            print('must be a positive integer > 0')
        self.k_clusters = kClusters
        return self.k_clusters
    
    #create a training and test set of the data. Timeseries will need to be handled
    #differently to other data..
    def create_training_test_set(self, ratio=.8, timeseries=False):
        if ratio < 0.0 or ratio > 1.0:
            print('ratio must be as a float between 0.0 and 1.0')
            return False
        print('creating a training and test dataset with a ratio of {}:{}'.format(ratio, 1-ratio))
        self.training_set, self.test_set = train_test_split(self.point_cloud, ratio)
        if self.training_set is not None and self.test_set is not None:
            return True
        return False
        pass

    #assign k clusters to list
    def initialise_clusters(self):
        print('initialising clusters')
        self.init_strategy.init(self.k_clusters, self.training_set)
        #Returns a list of indices of the initial cluster points of the dataset

    def initial_observations(self):
        print('running initial observation of clusters...')
        #TODO
        pass

    def recursive_observations(self):
        print('running algorithm...')
        #TODO
        pass

    def print_clusters(self):
        print('Here are the clusters')
        #Lorenz: Clusters are thousands of dots, printing them to command line might not be the ideal option for visualizing them.
        #TODO
        pass



#If we want to run as a script using some test data
if __name__ == '__main__':
    kmeans = KMeans('data/Skin_NonSkin.txt')
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.dataSummary()
    kmeans.initialise_clusters()
    kmeans.initial_observations()
    kmeans.recursive_observations()
    kmeans.print_clusters()
    kmeans.visualize_clusters_skin_noskin()
        