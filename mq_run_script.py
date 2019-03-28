from kmeans import KMeans
from init_strategies import PreClusteredSampleInit
from init_strategies import FarthestPointsInit
from init_strategies import RandomInit
from update_strategies import LloydUpdate, MacQueenUpdate

def setup_txt():
    kmeans = KMeans('data/Skin_NonSkin.txt', 3)
    kmeans.import_data()
    kmeans.convert_data()
    return kmeans

def setup_csv():
    kmeans = KMeans('data/HTRU2/HTRU_2.csv', 3)
    kmeans.import_data()
    kmeans.convert_data()
    return kmeans

kmeans=setup_txt()
kmeans.transform_skin_noskin_data()
init_strategy = RandomInit()
kmeans.init_centroids = init_strategy.init(k_clusters=2, point_cloud=kmeans.transformed_point_cloud)

def macqueen():
    kmeans.update_strategy = MacQueenUpdate()
    kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.transformed_point_cloud)   
    kmeans.visualize_clusters_skin_noskin()

def lloyd():
    kmeans.update_strategy = LloydUpdate()
    # kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
    kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)
    kmeans.visualize_clusters_skin_noskin()
        
        

# finalLloyd = lloyd()
macqueen()
# print(len(finalLloyd[0]))
# for i in range(len(finalLloyd[0])):
#     for j in range(len(finalLloyd[0])):
#         print(finalLloyd[0][i][j], finalMacqueen[i][j])