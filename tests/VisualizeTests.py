import unittest
import kmeans
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate


class VisualizeTests(unittest.TestCase):

    def setup_txt(self):
        kmeans = KMeans('../data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        return kmeans

    def setup_csv(self):
        kmeans = KMeans('../data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
        return kmeans

    def test_visualise_three_test(self):
        kmeans = self.setup_txt()
        kmeans.visualise_clusters()
        pass

    def visualise_ten_test(self):
        pass

    def visualise_none_test(self):
        pass

    def no_visualise_test(self):
        pass

    def corrupt_data_test(self):
        pass