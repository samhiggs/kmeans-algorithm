from abc import ABC, abstractmethod
import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import math

'''
Implementation of different initialization strategies
using the strategy pattern.
'''

class AbstractInit(ABC):

    #Parameters: number of clusters, data set
    #Return indices of initial cluster points
    @abstractmethod
    def init(self, k_clusters, point_cloud):
        pass

class RandomInit(AbstractInit):

    def init(self, k_clusters, point_cloud):
        pass

#See the following sources for description of the algorithm, particularly (1)
#(1)Page 17: http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
#(2)https://larssonjohan.com/post/2016-10-30-farthest-points/
class FarthestPointsInit(AbstractInit):
    def init(self, k_clusters, point_cloud):

        point_cloud_working_set = np.array(point_cloud)
        centroids_indices = []

        #First pick a random point
        np.random.seed(2019)
        centroids_indices.append(np.random.randint(low = 0, high = len(point_cloud)-1))

        #WHILE there are fewer than k points DO
        #   Add the point whose minimum distance from the selected points is as large as possible;
        #END
        while len(centroids_indices) < k_clusters:
            distances = []
            for i in range(0, len(centroids_indices)):
                for j in range(0, len(point_cloud)):
                    dist = np.linalg.norm(point_cloud[centroids_indices[i]] - point_cloud[j], ord=None)
                    distances.append((j, dist))

            min_dist = float("inf")
            ind = None
            for tuple in distances:
                if tuple[1] < min_dist:
                    min_dist = tuple[1]
                    ind = tuple[0]

            point_cloud = np.delete(point_cloud, point_cloud[ind], 0)
            centroids_indices.append(ind)
            print(max_dist)
            print(centroids_indices)
            pass

class PreClusterdSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self, k_clusters, point_cloud):
        pass