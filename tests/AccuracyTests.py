import unittest
from kmeans import KMeans
# import KMeans
# from init_strategies import *

class AccuracyTests(unittest.TestCase):

    # def withinClusterSumOfSquaresTest(self):
    #     pass

    # def outputResult(self):
    #     pass

    # def setup_txt(self):
    #     kmeans = KMeans('../data/Skin_NonSkin.txt', 2)
    #     kmeans.import_data()
    #     kmeans.convert_data()
    #     kmeans.transform_skin_noskin_data()
    #     init_strategy = PreClusteredSampleInit()
    #     kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    #     kmeans.update_strategy = LloydUpdate()
    #     kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    #     return kmeans

    # def setup_csv(self):
    #     kmeans = KMeans('../data/HTRU2/HTRU_2.csv', 2)
    #     kmeans.import_data()
    #     kmeans.convert_data()
    #     kmeans.transform_HTRU_data()
    #     init_strategy = PreClusteredSampleInit()
    #     kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    #     kmeans.update_strategy = LloydUpdate()
    #     kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    #     return kmeans

    def test_nmi_skin_noskin(self):
        print('running NMI TEST')
        # kmeans = self.setup_txt()
        kmeans = KMeans('',2)
        kmeans.raw_data = [0,1,2,3,4,5,6,7,8,9,10]
        kmeans.optimized_clusters = {
            0:['', [1,3,5,6,7,9]],
            1:['', [0,2,4,6,8,10]]
        }
        kmeans.true_result_dict = { 
            0: [1,3,5,7,8],
            1: [0,2,4,6,9,10]
        }
        nmi = kmeans.nmi_comparison()
        # nmi = kmeans.calc_nmi_skin_noskin_data()
        # print(nmi)
        assert nmi is not None
        assert nmi <= 1.0
        assert nmi >= 0.0
        pass

    # def test_wscc_skin_noskin(self):
    #     kmeans = self.setup_txt()
    #     wcss = kmeans.calc_wcss()
    #     print(wcss)
    #     assert wcss is not None
    #     assert wcss > 0.0
    #     pass

    # def test_wscc_HTRU(self):
    #     kmeans = self.setup_csv()
    #     wcss = kmeans.calc_wcss()
    #     print(wcss)
    #     assert wcss is not None
    #     assert wcss > 0.0
    #     pass

    # def test_nmi_HTRU(self):
    #     kmeans = self.setup_csv()
    #     nmi = kmeans.calc_nmi_HTRU_data()
    #     print(nmi)
    #     assert nmi is not None
    #     assert nmi <= 1.0
    #     assert nmi >= 0.0
    #     pass