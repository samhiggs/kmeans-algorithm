import kmeans
import time
import numpy as np
from kmeans import KMeans
from init_strategies import FarthestPointsInit
from init_strategies import PreClusteredSampleInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate
from update_strategies import MacQueenUpdate

def SkinRandomInitLloydUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = RandomInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRURandomInitLloydUpdate():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_HTRU_data()
    init_strategy = RandomInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()



def SkinFarthestPointsInitLloydUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = FarthestPointsInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRUFarthestPointsInitLloydUpdate():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_HTRU_data()
    init_strategy = FarthestPointsInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()




def SkinPreClusteredSampleInitLloydUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = PreClusteredSampleInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRUPreClusteredSampleInitLloydUpdate():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_HTRU_data()
    init_strategy = PreClusteredSampleInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = LloydUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()








def SkinRandomInitMacQueenUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = RandomInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRURandomInitMacQueenUpdate():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_HTRU_data()
    init_strategy = RandomInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()



def SkinFarthestPointsInitMacQueenUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = FarthestPointsInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRUFarthestPointsInitMacQueenUpdate():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_HTRU_data()
    init_strategy = FarthestPointsInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_HTRU_data(), kmeans.calc_wcss()




def SkinPreClusteredSampleInitMacQueenUpdate():
    kmeans = KMeans('data/Skin_NonSkin.txt', 2)
    kmeans.import_data()
    kmeans.convert_data()
    kmeans.transform_skin_noskin_data()
    init_strategy = PreClusteredSampleInit()
    kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)
    return kmeans.calc_nmi_skin_noskin_data(), kmeans.calc_wcss()

def HTRUPreClusteredSampleInitMacQueenUpdate():
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
    strategiesList = [SkinRandomInitLloydUpdate, HTRURandomInitLloydUpdate,
                      SkinFarthestPointsInitLloydUpdate, HTRUFarthestPointsInitLloydUpdate,
                      SkinPreClusteredSampleInitLloydUpdate, HTRUPreClusteredSampleInitLloydUpdate]

                     # SkinRandomInitMacQueenUpdate, HTRURandomInitMacQueenUpdate,
                     # SkinFarthestPointsInitMacQueenUpdate, HTRUFarthestPointsInitMacQueenUpdate,
                     # SkinPreClusteredSampleInitMacQueenUpdate, HTRUPreClusteredSampleInitMacQueenUpdate]

    timeDataRecords = np.zeros((6, 3, 100))
    for strategyType in range(6):
        for iterations in range(1):
            startTime = time.perf_counter()
            nmi, wcss = strategiesList[strategyType]()
            resultTime = time.perf_counter() - startTime
            timeDataRecords[strategyType, 0, iterations] = nmi
            timeDataRecords[strategyType, 1, iterations] = wcss
            timeDataRecords[strategyType, 2, iterations] = resultTime

    result = np.zeros((6, 3))
    for strategyType in range(6):
        result[strategyType, 0] = np.average(timeDataRecords[strategyType, 0])
        result[strategyType, 1] = np.average(timeDataRecords[strategyType, 1])
        result[strategyType, 2] = np.average(timeDataRecords[strategyType, 2])


    pass