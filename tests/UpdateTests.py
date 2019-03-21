import unittest
from kmeans import KMeans
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


    def test_lloyd_update_csv(self):
        kmeans=self.setup_csv()
        #init_strategy = RandomInit()
        init_strategy = FarthestPointsInit()
        kmeans.clusters = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.update_strategy.update(kmeans.clusters, kmeans.point_cloud)
        #TODO: find assert condition. Maybe evaluate using the measure of compactness and quality of the clsuter from the slides
        #assert len(kmeans.clusters) == kmeans.k_clusters
        pass