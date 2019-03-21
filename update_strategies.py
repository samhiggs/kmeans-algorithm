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
        centroids = []
        for i in centroid_indices:
            centroids.append(point_cloud[i])

        for j, centroid in enumerate(centroids):
            clusters[j] = [centroid,[]] #j = cluster id, centroid is the clusters centroid, [] cointains the indices of the points of the cluster

        modified = False

        #print(centroids)
        iter = 0
        while not modified:
            print('\n\nIteration: {}'.format(iter))
            iter += 1
            #Calculate new cluster
            for key in clusters.keys():
                clusters[key][1]=[]

            for k, point_idx in enumerate(point_cloud):
                min_dist = float("inf")
                #for l, centroid in enumerate(centroids):
                for key in clusters.keys():
                    dist = np.linalg.norm(clusters[key][0]- point_idx, ord=None)
                    if min_dist > dist:
                        min_dist = dist
                        clusters[key][1].append(k)

            #Reposition Centroids
            for key in clusters.keys():
                avg = [0]*len(point_cloud[0])
                for point_idx in clusters[key][1]:
                    avg += point_cloud[point_idx]
                avg = (1/len(clusters[key][1]))*avg
                print(avg)
                clusters[key][0] = avg


        modified = True
        print(clusters)
class MacQueenUpdate(AbstractUpdate):
    def update(self, centroid_indices, point_cloud):
        pass