import unittest
from kmeans import KMeans

class LoadDataTests(unittest.TestCase):

    def setupTxt(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/Skin_NonSkin.txt', 3)
        kmeans.import_data()
        return kmeans

    def setupCsv(self):
        kmeans = KMeans('/home/lorenz/PycharmProjects/sdm_kmeans/data/HTRU2/HTRU_2.csv', 3)
        kmeans.import_data()
        return kmeans

    def test_import_data_basic_txt(self):
        kmeans = self.setupTxt()
        assert kmeans.raw_data is not None
        pass

    def test_import_data_basic_csv(self):
        kmeans = self.setupCsv()
        assert kmeans.raw_data is not None
        pass

    def test_import_data_columns_txt(self):
        kmeans = self.setupTxt()
        assert len(kmeans.raw_data.columns) > 0

        kmeans = self.setupCsv()
        assert len(kmeans.raw_data.columns) > 0
        pass

    def test_import_data_columns_csv(self):
        kmeans = self.setupCsv()
        assert len(kmeans.raw_data.columns) > 0
        pass

    def test_convert_data_pointlist_txt(self):
        kmeans = self.setupTxt()
        kmeans.convert_data()
        assert len(kmeans.point_cloud) == len(kmeans.raw_data)
        pass

    def test_convert_data_pointlist_csv(self):
        kmeans = self.setupCsv()
        kmeans.convert_data()
        assert len(kmeans.point_cloud) == len(kmeans.raw_data)
        pass