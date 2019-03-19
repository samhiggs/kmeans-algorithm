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

#See the following sources for description of the algorithm, particularly (1),(2)
#(1)Page 17: http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
#(3)Page 3: https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
#(2)https://larssonjohan.com/post/2016-10-30-farthest-points/
class FarthestPointsInit(AbstractInit):
    def init(self, k_clusters, point_cloud):

        point_cloud_working_set = np.array(point_cloud)
        centroids_indices = []

        #First pick a random point
        np.random.seed(2019)
        centroids_indices.append(np.random.randint(low = 0, high = len(point_cloud)-1))

        #While the number of centroids is smaller than the number of desired clusters
        while len(centroids_indices) < k_clusters:

            #For each point that is not a centroid, find its distances to all centroids and push into a KV dictionary.
            #Key = point index, Values = array of distances to the centroids
            distances = {}
            for point_idx in range(0, len(point_cloud)):
                for centroid_idx in range(0, len(centroids_indices)):
                    dist = np.linalg.norm(point_cloud[centroids_indices[centroid_idx]] - point_cloud[point_idx], ord=None)
                    if not centroids_indices.__contains__(point_idx):
                        if point_idx in distances:
                            currently_stored = distances.get(point_idx)
                            distances[point_idx] = np.append(currently_stored, dist)
                        else:
                            distances[point_idx] = [dist]

            #For each points array of distances to all centroids, find its smallest distance to a centroid
            min_distances = {}
            distance_keys = distances.keys()
            for key in distance_keys:
                distances_per_point = distances.get(key)
                min_distance_per_point = float("inf")
                for distance in distances_per_point:
                    if distance < min_distance_per_point:
                        min_distance_per_point = distance
                min_distances[key] = min_distance_per_point

            #From the array of smallest distancees to any centroid, find the largest one.
            #The point that has this largest distance will become the next centroid
            min_distance_keys = min_distances.keys()
            max_dist = 0
            ind = None
            for key in min_distance_keys:
                dist = distances.get(key)
                if dist[0] > max_dist:
                    max_dist = dist[0]
                    ind = key

            #Add the new index to the centroids
            centroids_indices.append(ind)
            print(max_dist)
            print(centroids_indices)
            pass

class PreClusterdSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self, k_clusters, point_cloud):
        pass