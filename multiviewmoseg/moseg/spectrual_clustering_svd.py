import numpy as np
from scipy.linalg import svd
from scipy.sparse import eye as speye
from sklearn.cluster import KMeans


def spectral_clustering_svd(
    K: np.ndarray,
    num_groups: int,
    method: str = "normalized",
    max_iter: int = 1000,
    kmeans_replication: int = 100,
):
    """
    K: NxN adjacency matrix
    num_groups: number of groups for segmentation
    method: ['unnormalized', 'randomwalk', 'normalized']
    """
    n = K.shape[0]

    assert method in ["unnormalized", "randomwalk", "normalized"]

    groups, singular_values, laplacian = None, None, None

    if method == "unnormalized":
        D = np.diag(K.sum(axis=1))
        L = D - K
        _, S, V = svd(L)
        kernel = V[:, -num_groups:]

    elif method == "randomwalk":
        D = np.diag(1 / K.sum(axis=1))
        L = speye(n) - D @ K
        _, S, V = svd(L)
        kernel = V[:, -num_groups:]

    elif method == "normalized":
        D = np.diag(1 / np.sqrt(K.sum(axis=1)))
        L = speye(n) - D @ K @ D
        _, S, V = svd(L)
        kernel = V[:, -num_groups:]
        kernel = kernel / np.linalg.norm(kernel, axis=1, keepdims=True)

    else:
        raise NotImplementedError

    singular_values = S
    kmeans = KMeans(
        n_clusters=num_groups,
        init="k-means++",
        n_init=kmeans_replication,
        max_iter=max_iter,
        random_state=0,
    )
    groups = kmeans.fit_predict(kernel)
    laplacian = kernel

    return groups, singular_values, laplacian
