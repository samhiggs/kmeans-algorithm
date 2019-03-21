import unittest
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate

class UpdateTests(unittest.TestCase):

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


    def test_lloyd_update_txt(self):
        kmeans=self.setup_txt()
        #init_strategy = RandomInit()
        init_strategy = FarthestPointsInit()
        kmeans.clusters = init_strategy.init(k_clusters=4, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.update_strategy.update(kmeans.clusters, kmeans.point_cloud)
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass