import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def get_kernel_mask(labels, embeddings):
    '''
    Kernel: Pairwise squared euclidean distances, shape: (N, N), N - number of images
    mask: Same shape as Kernel, True if pair of images have the same label

    embeddings: N x d normalized image embeddings
    labels: labels associated with image / embeddings
    '''
    sorted_labels = labels[np.argsort(labels)]
    enc = OrdinalEncoder()
    enc.fit(sorted_labels.reshape(-1, 1))
    categories = enc.transform(sorted_labels.reshape(-1, 1))
    mask = (categories - categories.T == 0)

    emb_sorted = embeddings[np.argsort(labels)]
    X_norm_sq = np.sum(emb_sorted**2, axis=1)
    kernel = X_norm_sq[:, np.newaxis] + X_norm_sq[np.newaxis, :] - 2*np.dot(emb_sorted, emb_sorted.T)

    return kernel, mask


def val(K, mask, d, include_diag=True):
    '''
    VAL - Validation rate
    P_same: all possible pairs of images where labels are same
    TA: count of pairs from P_same where squared_distance between embeddings is less than d
    VAL: TA/count(P_same)
    '''
    TA = K[np.where(mask)] < d
    if include_diag:
        return np.sum(TA)/TA.shape[0]
    else:
        return (np.sum(TA) - N)/(TA.shape[0] - N)


def far(K, mask, d):
    '''
    FAR - False acceptance rate
    P_diff: pairs of images where labels are not same
    FA: count of pairs from P_diff where squared euclidean distance between embeddings is less than or equal to d
    FAR: FA/count(P_diff)
    '''
    FA = K[np.where(~mask)] <= d
    return np.sum(FA)/FA.shape[0]


def pairwise_accuracy(K, mask, d):
    '''
    TP: Count of all pairs where labels are same and pairwise squared euclidean distance is less than or equal to d
    TN: Count of all pairs where labels are not same and pairwise squared euclidean distance is greater than d
    '''
    N = K.shape[0]
    TP = np.sum(mask * (K <= d)) - N  # remove diagonal pairs (i,i)
    TN = np.sum((~mask) * (K > d))
    # print(TP, TN)
    return (TP+TN)/(N*(N-1))
