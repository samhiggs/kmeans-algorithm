from abc import ABC, abstractmethod
import numpy as np #useful for data analysis
import pandas as pd #useful for importing files and handling dataframes.
import math
import time
import operator
import sys
import threading

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
        print('\n\nInitializing with Random Points Strategy')

        seed = int(time.clock_gettime(time.CLOCK_REALTIME))
        np.random.seed(seed)
        print(seed)
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
        print('\n\nInitializing with Farthest Points Strategy')

        centroids_indices = []

        #First pick a random point.
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

class PreClusteredSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self, k_clusters, point_cloud):
        centroids_indices = []
        temp_centroids_indices = []
        clusters = []
        avg = 0
        seed = int(time.clock_gettime(time.CLOCK_REALTIME))
        np.random.seed(seed)
        avg_p = point_cloud[int(np.random.randint(low=0, high=len(point_cloud) - 1))]

        for point in point_cloud:
            avg += np.linalg.norm(avg_p-point, ord=None)

        avg /= len(point_cloud)

        step_size = avg/120

        rounds = 20
        seed = int(time.clock_gettime(time.CLOCK_REALTIME))
        np.random.seed(seed)
        # First pick a random point
        for i in range(0, k_clusters):
            temp_centroids_indices.append(int(np.random.randint(low=0, high=len(point_cloud) - 1)))
            clusters.append(list())


        for i in range(1, rounds-1):
            dup = 0
            for j, centroid in enumerate(temp_centroids_indices):
                new_points = self.points_in_range(centroid, point_cloud, step_size*i)
                for k in range(0, len(temp_centroids_indices)):
                    if k != j:
                        dup += self.check_duplicates(new_points, clusters[k])
                if dup > 0:
                    break
                clusters[j] = new_points
            print('Iteration: {}'.format(i))
            if dup > 0:
                break

        for cluster in clusters:
            centroids_indices.append(self.find_center(cluster, temp_centroids_indices, point_cloud))

        return centroids_indices

    pass


    def points_in_range(self, centroid, point_cloud, step_size):
        points_in_range = []
        for i, point in enumerate(point_cloud):
            if np.linalg.norm(point_cloud[centroid]-point, ord=None) <= step_size:
                points_in_range.append(i)
        return points_in_range

    def check_duplicates(self, new_points, other_list):
        dup = 0
        for new_point in new_points:
            for other_point in other_list:
                if new_point == other_point:
                    dup += 1

        return dup


    def find_center(self, cluster, centroids, point_cloud):
        center = None
        center_d = sys.maxsize
        for point in cluster:
            d_sum = 0
            if centroids.count(point) != 0:
                continue
            for other_point in cluster:
                d_sum += np.linalg.norm(point_cloud[other_point] - point_cloud[point], ord=None)

            if d_sum < center_d:
                center_d = d_sum
                center = point

        print('Center: {} und Distance_Sum: {}'.format(center, center_d))
        return center


