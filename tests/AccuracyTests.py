import unittest
import kmeans
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import PreClusteredSampleInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate

class AccuracyTests(unittest.TestCase):

    def withinClusterSumOfSquaresTest(self):
        pass

    def outputResult(self):
        pass

    def setup_txt(self):
        kmeans = KMeans('../data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans

    def setup_csv(self):
        kmeans = KMeans('../data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        return kmeans

    def test_nmi_skin_noskin(self):
        kmeans = self.setup_txt()
        nmi = kmeans.calc_nmi_skin_noskin_data()
        print(nmi)
        assert nmi is not None
        pass