import kmeans
import time
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import PreClusteredSampleInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate
from update_strategies import MacQueenUpdate

class CombineStrategies:
    def SkinRandomInitLloydUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRURandomInitLloydUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()



    def SkinFarthestPointsInitLloydUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRUFarthestPointsInitLloydUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()




    def SkinPreClusteredSampleInitLloydUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRUPreClusteredSampleInitLloydUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = LloydUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()








    def SkinRandomInitMacQueenUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRURandomInitMacQueenUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = RandomInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()



    def SkinFarthestPointsInitMacQueenUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRUFarthestPointsInitMacQueenUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = FarthestPointsInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()




    def SkinPreClusteredSampleInitMacQueenUpdate(self):
        kmeans = KMeans('data/Skin_NonSkin.txt', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_skin_noskin_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

    def HTRUPreClusteredSampleInitMacQueenUpdate(self):
        kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
        kmeans.import_data()
        kmeans.convert_data()
        kmeans.transform_HTRU_data()
        init_strategy = PreClusteredSampleInit()
        kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
        kmeans.update_strategy = MacQueenUpdate()
        kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
        return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()


if __name__ == '__main__':
    strategies = CombineStrategies()
