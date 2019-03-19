from abc import ABC, abstractmethod

'''
Implementation of different initialization strategies
using the strategy pattern.
'''

class AbstractInit(ABC):

    #Parameters: number of clusters, data set
    #Return indices of initial cluster points
    @abstractmethod
    def init(self, k_clusters, data):
        pass

class RandomInit(AbstractInit):

    def init(self, k_clusters, data):
        pass

class FarthestPointsInit(AbstractInit):
    #See page 17: http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    #https://larssonjohan.com/post/2016-10-30-farthest-points/
    def init(self, k_clusters, data):
        k_cluster_count = 0
        while k_cluster_count <= k_clusters:
            k_cluster_count+=1


        pass

class PreClusterdSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self, k_clusters, data):
        pass