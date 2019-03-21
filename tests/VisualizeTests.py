import unittest
import kmeans
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate


class VisualizeTests(unittest.TestCase):

    def setup_txt(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/Skin_NonSkin.txt', 3)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans

    def setup_csv(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/HTRU2/HTRU_2.csv', 3)
        kmeans.import_data()
        kmeans.convert_data()
        return kmeans

    def visualise_three_test(self):

        pass

    def visualise_ten_test(self):
        pass

    def visualise_none_test(self):
        pass

    def no_visualise_test(self):
        pass

    def corrupt_data_test(self):
        pass