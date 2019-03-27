import unittest
from kmeans import KMeans
from init_strategies import PreClusteredSampleInit
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate

class UpdateTests(unittest.TestCase):

    def setup_txt(self):
        kmeans = KMeans('../data/Skin_NonSkin.txt', 3)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans

    def setup_csv(self):
        kmeans = KMeans('../data/HTRU2/HTRU_2.csv', 3)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans


    def test_lloyd_update_random_init_csv(self):
        kmeans=self.setup_csv()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        for key in kmeans.optimized_clusters.keys():
            print(len(kmeans.optimized_clusters[key][1]))
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass

    def test_lloyd_update_farthest_init_csv(self):
        kmeans=self.setup_csv()
        #init_strategy = RandomInit()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        for key in kmeans.optimized_clusters.keys():
            print(len(kmeans.optimized_clusters[key][1]))
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass

    def test_lloyd_update_random_init_txt(self):
        kmeans=self.setup_txt()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        for key in kmeans.optimized_clusters.keys():
            print(len(kmeans.optimized_clusters[key][1]))
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass

    def test_lloyd_update_farthest_init_txt(self):
        kmeans=self.setup_txt()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        for key in kmeans.optimized_clusters.keys():
            print(len(kmeans.optimized_clusters[key][1]))
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass

    def test_lloyd_update_pre_clustered_sample_init_csv(self):
        kmeans=self.setup_csv()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass

    def test_lloyd_update_pre_clustered_sample_init_txt(self):
        kmeans=self.setup_txt()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass