import numpy as np

def chamfer_loss_numpy(y_true,y_pred):
    '''
    Calculate the chamfer distance,use euclidean metric
    :param y_true:
    :param y_pred:
    :return:
    '''
    y_true = np.reshape(y_true,[-1,num_pnts,3])
    y_pred = np.reshape(y_pred, [-1,num_pnts, 3])
    batch_size = y_true.shape[0]
    num_t = y_true.shape[1]
    num_p = y_pred.shape[1]
    dists_mat = np.zeros((num_t, num_p))
    _sum = 0.0
    loss_before_mean_py = []
    for bi in range(batch_size):
        for i in range(num_t):
            pnt_t = y_true[bi][i]
            for j in range(num_p):
                pnt_p = y_pred[bi][j]
                if (i <= j):
                    pnt_p = y_pred[bi][j]
                    dists_mat[i][j] = np.sum((pnt_t - pnt_p)**2)
                else:
                    dists_mat[i][j] = dists_mat[j][i]

        dist_t_to_p = np.mean(np.min(dists_mat, axis=0))
        dist_p_to_t = np.mean(np.min(dists_mat, axis=1))
        _sum += np.max([dist_p_to_t, dist_t_to_p])
        loss_before_mean_py.append(np.max([dist_p_to_t, dist_t_to_p]))
    return _sum / batch_size
    
num_pnts = 8
np.random.seed(1)
Y_true = np.random.randn(32, num_pnts, 3).astype(np.float32)
Y_pred = np.random.randn(32, num_pnts, 3).astype(np.float32)

loss_py = chamfer_loss_numpy(Y_true,Y_pred)
print(loss_py)
