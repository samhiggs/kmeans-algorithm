import unittest
from kmeans import KMeans
from init_strategies import FarthestPointsInit

class InitializationTests(unittest.TestCase):

    def setup(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/Skin_NonSkin.txt', 3)
        kmeans.importData()
        kmeans.convertData()
        return kmeans

    def test_farthest_points_init(self):
        kmeans=self.setup()
        initStrategy = FarthestPointsInit()
        initStrategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)

    def test_cluster_create(self):

        pass

    def test_invalid_cluster(self):
        pass

    def test_invalid_inputs(self):
        pass