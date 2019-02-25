import numpy as np
from sklearn.neighbors import KDTree

def chamfer_distance_sklearn(array1,array2):
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist

batch_size = 8
num_point = 20
num_features = 4
np.random.seed(1)
array1 = np.random.randint(0, high=4, size=(batch_size, num_point, num_features))
array2 = np.random.randint(0, high=4, size=(batch_size, num_point, num_features))

print('sklearn: ', chamfer_distance_sklearn(array1, array2))
