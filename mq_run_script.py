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

print('running mq update')
kmeans=setup_csv()
init_strategy = RandomInit()
kmeans.init_centroids = init_strategy.init(k_clusters=3, point_cloud=kmeans.point_cloud)
kmeans.update_strategy = MacQueenUpdate()
print('test macqueens about to start clustering')
kmeans.optimized_clusters = kmeans.update_strategy.update(kmeans.init_centroids, kmeans.point_cloud)