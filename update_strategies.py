from abc import ABC, abstractmethod
import numpy as np #useful for data analysis
import time
import copy
class AbstractUpdate(ABC):

    #Receives self, # of clusters, list of indices of initial centroids, and point cloud as parameters
    #Returns a KV dict containing the centroid_id as key, the centroids and the an array of point indices as values
    def update(self, centroids, point_cloud, model_metadata):
        pass

#Source:
#https://www.cs.utah.edu/~jeffp/teaching/cs5955/L10-kmeans.pdf
#https://pdfs.semanticscholar.org/0074/4cb7cc9ccbbcdadbd5ff2f2fee6358427271.pdf
class LloydUpdate(AbstractUpdate):

    def update(self, centroid_indices, point_cloud, model_metadata):
        print('\n\nUpdating with Lloyd Update Strategy')
        start = time.time()
        #Initialize cluster KV dictionary, centroids.
        clusters = {}
        n_iterations = 0
        for i in centroid_indices:
            clusters[i] = [point_cloud[i],[]]  # i = cluster id, centroid is the clusters centroid, [] contains the indices of the points of the cluster
        iter = 0
        n_iterations += 1
        modified = True
        while modified:
            # print('\nIteration: {}'.format(iter))
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
                # print(avg)
                diff_centroids.append(clusters[key][0] - avg)
                clusters[key][0] = avg

            #Check for abort condition:
            eps = 10**-6
            # print('Centroids moved by: {}'.format(diff_centroids))
            modified = False
            for centroid in diff_centroids:
                for elem in centroid:
                    if eps < np.linalg.norm(elem):
                        modified = True
                        break
                if modified: break
        end = time.time()
        print('Lloyds completed in {0:.2f} time.'.format(end - start))
        print('Total iterations: {}'.format(n_iterations))
        print('final centroids: ')
        model_metadata['speed'] = end-start
        model_metadata['n_iterations'] = n_iterations
        model_metadata['n_updates'] = 0
        model_metadata['average_time_per_update'] = (end-start)/n_iterations
        model_metadata['n_clusters'] = len(clusters.keys())
        model_metadata['clusters'] = {} 
        
        # for k,v in clusters.items():
        #     model_metadata['clusters'][k] = {
        #         'n_points': len(v['point_indices']),
        #         'centroid': v['centroid']
        #     }
        return clusters

class MacQueenUpdate(AbstractUpdate):
    def update(self, centroid_indices, point_cloud, model_metadata):

        print('\n\nUpdating with Macqueens Update Strategy')
        start = time.time()
        n_updates = 0
        sum_update_time = 0.0
        n_iterations = 0
        epsilon = [10**-6]*len(point_cloud[0])
        meaningful_diff = True
        prevClusters = {}
        updated_centroids = [point_cloud[i] for i in centroid_indices]
        mean_dist_per_iteration = {}
        while True:
            # Base conditions
            if(n_iterations >= 10):
                break
            if not meaningful_diff:
                break

            n_updates_per_iteration = 0
            iter_start = time.time()
            tmpClusters = {}
            for i,v in enumerate(updated_centroids):
                # Check if there has been a meaningful change in data since the last pass
                update_start = time.time()
                tmpClusters[i] = {
                    'centroid': v,
                    'point_indices': {},
                    'sumOfPoints': v
                }

            for idx, point in enumerate(point_cloud):
                
                min_dist = float("inf")
                dist = -1
                cluster = -1
                for clusterIdx, values in tmpClusters.items():
                    dist = np.linalg.norm(values['centroid'] - point, ord=None)
                    if min_dist > dist:
                        min_dist = dist
                        cluster = clusterIdx
                # print(min_dist, cluster, dist)
                try:
                    clusterData = tmpClusters[cluster]
                except:
                    print('Key error for: ', n_iterations, idx)
                clusterData['point_indices'][idx] = dist
                
                newSum = np.sum([clusterData['sumOfPoints'], point], axis=0)
                newCentroid = np.divide(newSum, len(clusterData['point_indices']))

                clusterData.update({'sumOfPoints': newSum})
                if not np.allclose(newCentroid, clusterData['centroid'], epsilon):
                    clusterData.update({'centroid': newCentroid}) 
                    n_updates_per_iteration += 1
                    meaningful_diff = True
                    idx_of_last_change = i

                update_end = time.time()
                sum_update_time += update_end - update_start

            mean_dist_per_iteration[n_iterations] = {}
            for k,v in tmpClusters.items():
                #Calculate the averages to see how well the run did
                mean_dist_per_iteration[n_iterations][k] = {
                        'avg_dist' : v['centroid'],
                        'mean_of_points': np.mean(v['centroid'])
                }
            updated_centroids = [v['centroid'] for k,v in tmpClusters.items()]
            if n_iterations > 1:
                try:
                    finishedList = [False for k,v in tmpClusters.items() if not np.allclose(prevClusters[k]['centroid'], tmpClusters[k]['centroid'], epsilon)]
                    if False not in finishedList:
                        meaningful_diff = False          
                except:
                    print('key error', prevClusters.keys(), tmpClusters.keys())
            prevClusters = tmpClusters
            iter_end = time.time()
            print('\nIteration: {}, time {}, updates: {}'
                .format(n_iterations, iter_end - iter_start, n_updates_per_iteration))
            n_updates += n_updates_per_iteration
            n_iterations += 1


        end = time.time()
        #Meta data for the 
        print('Macqueens completed in {:.2f} seconds'.format(end - start))
        print('Total iterations: {}'.format(n_iterations))
        print('Total updates: {}, Average update time {}'.format(n_updates, sum_update_time/n_updates))
        print('final centroids: ')
        model_metadata['speed'] = end-start
        model_metadata['n_iterations'] = n_iterations
        model_metadata['n_updates'] = n_updates
        model_metadata['average_time_per_update'] = sum_update_time/n_updates
        model_metadata['n_clusters'] = len(prevClusters.keys())
        model_metadata['clusters'] = {} 
        
        for k,v in prevClusters.items():
            model_metadata['clusters'][k] = {
                'n_points': len(v['point_indices']),
                'centroid': v['centroid']
            }

            print('cluster {}:'.format(k))
            [print('    ',i, ': ', val) for i,val in v.items() if i != 'point_indices']
            print('Number of points: {}'.format(len(v['point_indices'])))
            # print('Point distances:')
            # for p in v['point_indices']:
            #     print('cluster: ', k, ' idx: ', p, '  distances: ', 
            #     [print('cluster: ', k, 'dist', np.linalg.norm(point_cloud[p] - v['centroid'], ord=None)) for k,v in tmpClusters.items()])
        print('\n')
        for k,v in mean_dist_per_iteration.items():
            print('Iteration ', k)
            for subK, subV in v.items():
                print('cluster:', subK)
                print('mean of centroid points', subV['mean_of_points'])
        
        finalPoints = {}
        for k,v in prevClusters.items():
            finalPoints[k] = [v['centroid'], list(v['point_indices'].keys())]
        return finalPoints
