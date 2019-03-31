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

@param k_clusters
@param init_centroids = []
@param optimized_clusters = {}
@param raw_data is the data that comes in from the csv or text file
@param processed_data is the result of any processing, such as removing result column.
@param true_result_dict converts the true results into a dictionary
@param raining_set not in use
        self.test_set not in use
        self.init_strategy defines the initialisation stragey used
        self.update_strategy defines the update strategy used
'''

import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import sklearn as skl #useful for initial analysis
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from init_strategies import PreClusteredSampleInit, FarthestPointsInit, RandomInit
from update_strategies import LloydUpdate, MacQueenUpdate
from scipy import stats

import math
import seaborn as sns #useful for splitting test and training data set and other machine learning methods
from matplotlib import pyplot as plt #useful for visualising data
import abc
import sys
import configparser as cp
import hashlib
import itertools
from datetime import datetime as dt
import time
import csv
from mpl_toolkits import mplot3d

import os

class KMeans:
    
    def __init__(self, filename='', k_clusters=0, strategies=None, dataset=None):
        self.filename = filename
        self.k_clusters = k_clusters
        self.init_centroids = []
        self.optimized_clusters = {}
        self.model_metadata = {}
        self.raw_data = dataset
        self.processed_data = dataset
        self.true_result_dict = None
        self.init_strategy = None
        self.update_strategy = None
        self.plot_figure = None
        self.results_dir = None
        self.function_map = {
            'RandomInit': RandomInit,
            'FarthestPointsInit': FarthestPointsInit,
            'PreClusteredSampleInit' : PreClusteredSampleInit,
            'MacQueenUpdate': MacQueenUpdate,
            'LloydUpdate': LloydUpdate
        }
        self.function_runtime_data = {
            'import_data': [],
            'visualize_clusters' : [],
            'process_true_data' : [],
            'nmi_comparison' : [],
            'calc_wcss' : [],
            'export_results' : [],
            'import_results' : [],
        }
    #imports raw data and checks that it is a valid filetype.
    def import_data(self):
        start = time.time()
        print('importing data from {}'.format(self.filename))
        #read the filetype and run relevant case
        filename, ftype = os.path.splitext(self.filename)
        if ftype == '.csv':
            self.raw_data = np.genfromtxt(self.filename, delimiter=',', dtype=float, usecols=(0,1,2,3,4,5,6,7,8))
        elif ftype == '.txt':
            self.raw_data = np.genfromtxt(self.filename, delimiter="\t", dtype=int, usecols=(0,1,2,3))
        else:
            print('{} is an invalid filetype'.format(ftype))
        #Remove duplicates -> screws NMI, dont know why yet
        #self.data = np.unique(self.data, axis=0)
        #Normalize data -> screws WSCC
        #self.transformed_data = preprocessing.normalize(self.transformed_data)
        self.processed_data = self.raw_data[:,:-1]
        end = time.time()
        self.function_runtime_data['import_data'].append([end-start, len(self.raw_data), ftype])
        
    def visualize_clusters_skin_noskin(self):
        #Source: https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
        start = time.time()
        ax = plt.axes(projection='3d')

        # Data for three-dimensional scattered points
        zdata = []
        xdata = []
        ydata = []
        #self.data = np.arccosh(self.data)
        for i,key in enumerate(self.optimized_clusters.keys()):
            zdata.clear()
            xdata.clear()
            ydata.clear()
            for point in self.optimized_clusters[key][1]:
                zdata.append(self.raw_data[point][2])
                ydata.append(self.raw_data[point][1])
                xdata.append(self.raw_data[point][0])

            if i == 0:
                ax.scatter3D(xdata, ydata, zdata, c='r', marker='.')
            if i == 1:
                ax.scatter3D(xdata, ydata, zdata, c='b', marker='.')
            if i == 2:
                ax.scatter3D(xdata, ydata, zdata, c='g', marker='.')
            print(len(self.optimized_clusters[key][1]))
        fn = dt.now().strftime("%Y%m%d-%H%M%S")+'.png'
        if self.results_dir is not None:
            fn = self.results_dir + '/' + fn
        plt.savefig(fn)
        plt.show()
        #need to store figure
        end = time.time()
        self.function_runtime_data['visualize_clusters'].append(end-start)

    def process_true_data(self):
        start = time.time()
        print('reshaping true data in the form of results')
        result_col = np.shape(self.raw_data)[1]-1
        true_data_dict = {unique_result:[] for unique_result in np.unique(self.raw_data[:,result_col])}
        for idx, row in enumerate(self.raw_data):
            true_data_dict[row[result_col]].append(idx)
        self.true_result_dict = true_data_dict
        end = time.time()
        self.function_runtime_data['process_true_data'].append([end-start, len(true_data_dict), sys.getsizeof(true_data_dict)])

    def _cleanup(self, name):
        """Do cleanup for an attribute"""
        value = getattr(self, name)
        # self._pre_cleanup(name, value)
        delattr(self, name)
        # self._post_cleanup(name, value)
    
    def _cleanup_all(self):
        values = ['init_centroids', 'optimized_clusters', 'model_metadata',
            'init_strategy', 'update_strategy', 'plot_figure']
        for v in values:
            if v is not None:
                self._cleanup(v)
        for k in self.function_runtime_data.keys():
            self.function_runtime_data.update({k : []})
        print('attributes have been cleaned.')
        self.model_metadata = {}
        self.init_centroids = []
        self.optimized_clusters = {}
        self.init_strategy = None
        self.update_strategy = None
        self.plot_figure = None
    #Need the number of results. 
    def nmi_comparison(self):
        start = time.time()
        if self.true_result_dict is None:
            self.process_true_data()

        if(len(self.optimized_clusters.keys()) != len(self.true_result_dict)):
            print('The clusters or results have not been processed correctly.'\
                'There are {} model results and {} true results'
                .format(len(self.optimized_clusters.keys()), len(self.true_result_dict)))
            return
        #normalised scores per cluster
        normalised_scores = {}
        for cluster in self.true_result_dict.keys():
            normalised_scores[cluster] = metrics.cluster.normalized_mutual_info_score(
                        self.true_result_dict[cluster], 
                        self.optimized_clusters[cluster]
                    )
        self.model_metadata['nmi_comparison'] = normalised_scores

        end = time.time()
        self.function_runtime_data['nmi_comparison'].append([end-start, ])
        return normalised_scores

    def calc_wcss(self):
        start = time.time()
        wscc_results = {}
        for cluster in self.optimized_clusters.keys():
            wscc = 0.0
            centroid = self.optimized_clusters[cluster][0]
            points = self.optimized_clusters[cluster][1]
            for point_idx in points:
                wscc += np.power(np.linalg.norm(self.processed_data[point_idx] - centroid, ord=None),2)
            wscc_results[cluster] = [wscc, np.sqrt(wscc)]
            print(cluster, ':',wscc, 'and sd is:', np.sqrt(wscc))

        self.model_metadata['wscc'] = wscc
        print(wscc)
        end = time.time()
        self.function_runtime_data['calc_wcss'].append([end-start, sys.getsizeof(self.optimized_clusters)])
        return wscc

    def export_results(self, init_strategy, update_strategy):
        start = time.time()
        if self.optimized_clusters is None:
            print('no results yet. Try running the optimisation')
            return None
        # print(self.init_strategy, self.update_strategy)
        fn = dt.now().strftime("%Y%m%d-%H%M%S")+'.csv'
        if self.results_dir is not None:
            fnp = self.results_dir + '/' + \
                init_strategy.lower() + '-' + \
                update_strategy.lower() + '-' + fn
        with open(fnp, 'w', newline='') as csvfile:
            #write metadata to file
            writer = csv.writer(csvfile)
            writer.writerow(['runtime', dt.now().strftime("%Y%m%d-%H%M%S")])
            writer.writerow(['initialisation', 'TODO'])
            writer.writerow(['update', 'TODO'])
            for key, value in self.model_metadata.items():
                if type(value) == type(list()) or type(value) == type(dict()):
                    continue
                writer.writerow([key, value])
            #write cluster specific metadata
            if 'clusters' in self.model_metadata:
                writer = csv.writer(csvfile)
                for k,v in self.model_metadata['clusters'].items():
                    for ki, vi in v.items():
                        writer.writerow(['cluster ' + str(k), ki, vi])
            print(self.optimized_clusters.keys())
            #write cluster idx to file
            for kid, vid in self.optimized_clusters.items():
                writer.writerow([kid, vid[0]])
                for el in vid[1]:
                    writer.writerow([kid, el])

        if self.plot_figure is not None:
            #TODO
            pass

        end = time.time()

        self.function_runtime_data['export_results'].append([end-start, sys.getsizeof(self.optimized_clusters)])
        #Write function benchmarks to file
        bmarkfn = self.results_dir + '/runtime-data-'+fn
        with open(bmarkfn, 'w', newline='') as csvfile:
            bmarkWriter = csv.writer(csvfile)
            for bmark_key, bmark_val in self.function_runtime_data.items():
                bmarkWriter.writerow([bmark_key] + bmark_val)
    
    def import_results(self):
        #TODO
        pass

    #summary of data
    def dataSummary(self):
        #TODO
        if self.raw_data is None or len(self.raw_data) is 0:
            print('the raw_data has not been created in raw_dataSummary()')
            return False
        print('{}\n {}\n {}\n {}\n'.format(self.raw_data, self.raw_data.info(), self.raw_data.describe(), self.data.columns))
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
    # #differently to other data..
    # def create_training_test_set(self, ratio=.8, timeseries=False):
    #     if ratio < 0.0 or ratio > 1.0:
    #         print('ratio must be as a float between 0.0 and 1.0')
    #         return False
    #     print('creating a training and test dataset with a ratio of {}:{}'.format(ratio, 1-ratio))
    #     self.training_set, self.test_set = train_test_split(self.data, ratio)
    #     if self.training_set is not None and self.test_set is not None:
    #         return True
    #     return False
    #     pass

    #assign k clusters to list
    def initialise_clusters(self):
        print('initialising clusters')
        self.init_strategy.init(self.k_clusters, self.training_set)
        #Returns a list of indices of the initial cluster points of the dataset

#If we want to run as a script using some test data
if __name__ == '__main__':
    start = time.time()
    cf = cp.ConfigParser()
    try:
        cf.read('config_files/options.ini', )
    except:
        print('couldnt read config file')
        exit()

    if cf.sections() == 0:
        print('config is empty')
        exit()

    #Get data from parser    
    data_dir = cf['PATHS']['data_dir']
    results_dir = cf['PATHS']['results_dir']
    datasets = [(cf['DATASETS'][key]).split() for key in cf['DATASETS']]
    init_strategies = cf['STRATEGIES']['init_strategies'].split()
    update_strategies = cf['STRATEGIES']['update_strategies'].split()

    #clean the path and create an extension
    [d.append(os.path.splitext(d[0])[1]) for d in datasets]
    #Make sure it all looks good
    # print(data_dir, datasets, init_strategies, update_strategies)
    
    #Check dataset can be found, if not remove it.
    for dataset in datasets:
        exists = os.path.isfile(data_dir+'/'+dataset[0])
        if not exists:
            print(dataset, ' cannot be found')
            continue
        if  len(dataset) != 3:
            datasets.remove(dataset)
        try:
            dataset[1] = int(dataset[1])
        except:
            datasets.remove(dataset)

    if len(datasets) == 0:
        print('No datasets can be found')
        exit()
    s_combinations = [(i,j) for j in update_strategies for i in init_strategies]
    kmeans_instances = {}
    user_input = ''
    for i,dataset in enumerate(datasets):
        kmeans = KMeans(data_dir+'/'+dataset[0], dataset[1])
        kmeans.results_dir = results_dir
        kmeans.import_data()
        kmeans.process_true_data()
        while True:
            #If you want to run combinations independently, otherwise just comment out
            print('available combinations are:')
            [print(i, c) for i, c in enumerate(s_combinations)]
            print('Please select a combination or exit')
            user_input = input()
            if user_input == 'exit':
                break
            selection = int(user_input)
            kmeans.init_strategy = kmeans.function_map[s_combinations[selection][0]]()
            kmeans.update_strategy = kmeans.function_map[s_combinations[selection][1]]()
            kmeans.init_centroids = kmeans.init_strategy.init(k_clusters=dataset[1], point_cloud=kmeans.processed_data)
            kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.processed_data, kmeans.model_metadata)
            kmeans.calc_wcss()
            kmeans.export_results(s_combinations[selection][0], s_combinations[selection][1])
            kmeans._cleanup_all()
        # kmeans.nmi_comparison()

    kmeans_instances[i] = kmeans
    end = time.time()
    print('Program ran in {} seconds'.format(end-start))
    '''
    kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()
'''
    exit()
        