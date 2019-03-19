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

import math
import seaborn as sns #useful for splitting test and training data set and other machine learning methods
from matplotlib import pyplot as plt #useful for visualising data
import abc

class KMeans:
    
    def __init__(self, filename, k_clusters=0):
        self.filename = filename
        self.k_clusters = k_clusters
        self.clusters = []
        self.data = None
        self.training_set = None
        self.test_set = None
        self.init_strategy = None
        self.update_strategy = None

    #imports data and checks that it is a valid filetype
    def importData(self):
        print('importing data from {}'.format(self.filename))
        #read the filetype and run relevant case
        ftype = ''
        if ftype is 'csv':
            self.data = pd.read_csv("{}".format(self.filename))
        if ftype is 'txt':
            #TODO
            pass
        else:
            print('{} is an invalid filetype'.format(ftype))
            return

    #def outputTest
    #summary of data
    def dataSummary(self):
        #TODO
        if self.data is None or len(self.data) is 0:
            print('the data has not been created in dataSummary()')
            return False
        print('{}\n {}\n {}\n {}\n'.format(self.data.head(), self.data.info(), self.data.describe(), self.data.columns))
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
        self.training_set, self.test_set = train_test_split(self.data, ratio)
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
        #TODO
        pass

    def visualise_clusters(self):
        print('and in a nice pretty diagram!')
        #TODO
        pass


#If we want to run as a script using some test data
if __name__ == '__main__':
    kmeans = KMeans('data\Skin_NonSkin.txt')
    kmeans.importData()
    kmeans.dataSummary()
    kmeans.initialise_clusters()
    kmeans.initial_observations()
    kmeans.recursive_observations()
    kmeans.print_clusters()
    kmeans.visualise_clusters()
        