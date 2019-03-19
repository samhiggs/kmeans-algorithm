import unittest
from kmeans import KMeans

class LoadDataTests(unittest.TestCase):

    def setup(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/Skin_NonSkin.txt', 3)
        kmeans.importData()
        return kmeans

    def test_import_data_basic(self):
        kmeans = self.setup()
        assert kmeans.data is not None
        pass

    def test_import_data_columns(self):
        kmeans = kmeans = self.setup()
        assert len(kmeans.data.columns) > 0
        #print(kmeans.data[0])
        pass