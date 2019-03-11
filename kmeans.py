'''
#### Kmeans implementation assignment 1 for Scientific Data Management ####
### See test folder for tests
'''
import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import sklearn as skl #useful for initial analysis
import math
import seaborn as sns #useful for splitting test and training data set and other machine learning methods
from matplotlib import plt #useful for visualising data

class KMeans:
    
    def __init__(self, filename, k_clusters):
        self.filename = filename
        self.k_clusters = k_clusters
        self.clusters = []
        self.data = None

    #imports data and checks that it is a valid filetype
    def importData(self):
        self.data = pd.read_csv("{}".format(self.filename))

    #summary of data
    def dataSummary(self):
        #TODO
        if self.data is None or len(self.data) is 0:
            print('the data has not been created in dataSummary()')
            return
        print('{}\n {}\n {}\n {}\n'.format(self.data.head(), self.data.info(), self.data.describe(), self.data.columns))

    #assign k clusters to list
    def initalise_clusters(self):
        #TODO
        pass

    def initial_observations(self):
        #TODO
        pass

    def recursive_observations(self):
        #TODO
        pass

    def print_clusters(self):
        #TODO
        pass

    def visualise_clusters(self):
        #TODO
        pass
    