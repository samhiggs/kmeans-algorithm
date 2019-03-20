import unittest
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit

class InitializationTests(unittest.TestCase):

    def setup_txt(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/Skin_NonSkin.txt', 4)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans

    def setup_csv(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/HTRU2/HTRU_2.csv', 4)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans


    def test_farthest_points_init_txt(self):
        kmeans=self.setup_txt()
        init_strategy = FarthestPointsInit()
        kmeans.clusters = init_strategy.init(k_clusters=4, point_cloud=kmeans.point_cloud)
        assert len(kmeans.clusters) == kmeans.k_clusters

    def test_farthest_points_init_csv(self):
        kmeans=self.setup_csv()
        init_strategy = FarthestPointsInit()
        kmeans.clusters = init_strategy.init(k_clusters=4, point_cloud=kmeans.point_cloud)
        assert len(kmeans.clusters) == kmeans.k_clusters

    def test_random_points_init_txt(self):
        kmeans=self.setup_txt()
        init_strategy = RandomInit()
        kmeans.clusters = init_strategy.init(k_clusters=4, point_cloud=kmeans.point_cloud)
        assert len(kmeans.clusters) == kmeans.k_clusters

    def test_cluster_create(self):

        pass

    def test_invalid_cluster(self):
        pass

    def test_invalid_inputs(self):
        pass