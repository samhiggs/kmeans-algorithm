from abc import ABC, abstractmethod
import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import math
import time

'''
Implementation of different initialization strategies
using the strategy pattern.
'''

class AbstractInit(ABC):

    #Parameters: number of clusters, data set
    #Return a list containing the indices of initial cluster points
    @abstractmethod
    def init(self, k_clusters, point_cloud):
        pass

#Randomly initializes the k centroids for the cluster.
#TODO: Imporve defensive programming
class RandomInit(AbstractInit):

    def init(self, k_clusters, point_cloud):
        seed = int(time.clock_gettime(time.CLOCK_REALTIME))
        np.random.seed(seed)
        centroids_indices = []
        while len(centroids_indices) < k_clusters:
            centroids_indices.append(np.random.randint(low=0, high=len(point_cloud) - 1))

        #print(centroids_indices)
        return centroids_indices
        pass


#See the following sources for description of the algorithm, particularly (1),(2)
#(1)Page 17: http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
#(3)Page 3: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
#(2)https://larssonjohan.com/post/2016-10-30-farthest-points/
#TODO: Improve defensive programming
class FarthestPointsInit(AbstractInit):
    def init(self, k_clusters, point_cloud):

        centroids_indices = []

        #First pick a random point
        seed = int(time.clock_gettime(time.CLOCK_REALTIME))
        np.random.seed(seed)
        centroids_indices.append(np.random.randint(low = 0, high = len(point_cloud)-1))

        #While the number of centroids is smaller than the number of desired clusters
        while len(centroids_indices) < k_clusters:

            #For each point that is not a centroid, find its distances to all centroids.
            #Out of these distances take the smallest and push into a KV dictionary.
            #Key = point index, Value = smallest distances to the centroids
            #From the array of smallest distancees to any centroid, find the largest one.
            #The point that has this largest distance will become the next centroid
            min_distances = {}
            max_min_dist = 0
            new_centroid_idx = None
            for point_idx in range(0, len(point_cloud)):
                if not centroids_indices.__contains__(point_idx):
                    for centroid_idx in range(0, len(centroids_indices)):
                        dist = np.linalg.norm(point_cloud[centroids_indices[centroid_idx]] - point_cloud[point_idx], ord=None)
                        if point_idx in min_distances:
                            if min_distances.get(point_idx) > dist:
                                min_distances[point_idx] = dist
                        else:
                            min_distances[point_idx] = dist
                    if max_min_dist <min_distances.get(point_idx):
                        max_min_dist = min_distances.get(point_idx)
                        new_centroid_idx = point_idx

            centroids_indices.append(new_centroid_idx)
        return centroids_indices

class PreClusterdSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self, k_clusters, point_cloud):
        pass