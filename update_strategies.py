from abc import ABC, abstractmethod
import numpy as np #useful for data analysis

class AbstractUpdate(ABC):

    #Receives self, # of clusters, list of indices of initial centroids, and point cloud as parameters
    #Returns a list of lists containing the indices of the points in the point cloud.
    def update(self, centroids, point_cloud):
        pass

#Source:
#https://www.cs.utah.edu/~jeffp/teaching/cs5955/L10-kmeans.pdf
#https://pdfs.semanticscholar.org/0074/4cb7cc9ccbbcdadbd5ff2f2fee6358427271.pdf
class LloydUpdate(AbstractUpdate):

    def update(self, centroid_indices, point_cloud):
        print('\n\nUpdating with Lloyd Update Strategy')

        #Initialize cluster KV dictionary, centroids.
        clusters = {}
        for i in centroid_indices:
            #centroids.append(point_cloud[i])
            clusters[i] = [point_cloud[i],[]]  # j = cluster id, centroid is the clusters centroid, [] contains the indices of the points of the cluster

        iter = 0
        modified = True
        while modified:
            print('\n\nIteration: {}'.format(iter))
            iter += 1

            #Calculate new cluster
            for key in clusters.keys():
                clusters[key][1]=[]

            for k, point_idx in enumerate(point_cloud):
                min_dist = float("inf")
                best_key = None
                for key in clusters.keys():
                    dist = np.linalg.norm(clusters[key][0]- point_idx, ord=None)
                    if min_dist > dist:
                        min_dist = dist
                        best_key = key
                clusters[best_key][1].append(k)

            #Reposition Centroids
            diff_centroids = []
            for key in clusters.keys():
                avg = [0]*len(point_cloud[0])
                for point_idx in clusters[key][1]:
                    avg += point_cloud[point_idx]
                avg = (1/len(clusters[key][1]))*avg
                print(avg)
                diff_centroids.append(clusters[key][0] - avg)
                clusters[key][0] = avg

            #Check for abort condition:
            epsilon = [10**-6]*len(point_cloud[0])
            print('Centroids moved by: {}'.format(diff_centroids))
            modified = False
            for eps in epsilon:
                for centroid in diff_centroids:
                    for elem in centroid:
                        if eps < elem:
                            modified = True
                            break
                    if modified: break
                if modified: break
        return clusters

class MacQueenUpdate(AbstractUpdate):
    def update(self, centroid_indices, point_cloud):
        pass