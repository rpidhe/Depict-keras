from __future__ import print_function

import sys
import os
import time
import timeit
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
import tensorflow.keras.backend as K
try:
    import cPickle as pickle
except:
    import pickle
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
try:
    from six.moves import xrange
except:
    pass

import scipy
from numpy.matlib import repmat
from scipy.spatial.distance import cdist
from scipy import sparse


def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
    # Compute conditional complexity from the subpart of the weighted adjacency matrix
    # Inputs:
    #   - IminuszW: the matrix (I - z*P)
    #	- cluster_i: index vector of cluster i
    #	- cluster_j: index vector of cluster j
    # Output:
    #	- L_ij - the sum of conditional complexities of cluster i and j after merging.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    num_i = np.size(cluster_i)
    num_j = np.size(cluster_j)

    # detecting cross elements (this check costs much and is unnecessary)

    ijGroupIndex = np.append(cluster_i, cluster_j)

    y_ij = np.zeros((num_i + num_j, 2))  # [y_i, y_j]
    y_ij[:num_i, 0] = 1
    y_ij[num_i:, 1] = 1
    idx = np.ix_(ijGroupIndex, ijGroupIndex)
    L_ij = scipy.linalg.inv(IminuszW[idx]).dot(y_ij)
    L_ij = sum(L_ij[:num_i, 0]) / (num_i * num_i) + sum(L_ij[num_i:, 1]) / (num_j * num_j)

    return L_ij


def gacPathEntropy(subIminuszW):
    # Compute structural complexity from the subpart of the weighted adjacency matrix
    # Input:
    #   - subIminuszW: the subpart of (I - z*P)
    # Output:
    #	- clusterComp - strucutral complexity of a cluster.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    N = subIminuszW.shape[0]
    clusterComp = scipy.linalg.inv(subIminuszW).dot(np.ones((N, 1)))
    clusterComp = sum(clusterComp) / (N * N)

    return clusterComp


def gacMerging(graphW, initClusters, groupNumber, strDescr, z):
    # Cluster merging for Graph Agglomerative Clustering
    # Implements an agglomerative clustering algorithm based on maiximum graph
    #   strcutural affinity of two groups
    # Inputs:
    #	- graphW: asymmetric weighted adjacency matrix
    #   - initClusters: a cell array of clustered vertices
    #   - groupNumber: the final number of clusters
    #   - strDescr: structural descriptor, 'zeta' or 'path'
    #   - z: (I - z*P), default: 0.01
    # Outputs:
    #   - clusterLabels: 1 x m list whose i-th entry is the group assignment of
    #                   the i-th data vector w_i. Groups are indexed
    #                   sequentially, starting from 1.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    numSample = graphW.shape[0]
    IminuszW = np.eye(numSample) - z * graphW
    myInf = 1e10

    # initialization
    VERBOSE = True

    numClusters = len(initClusters)
    if numClusters <= groupNumber:
        print('GAC: too few initial clusters. Do not need merging!');

    # compute the structural complexity of each initial cluster
    clusterComp = np.zeros((numClusters, 1))
    for i in xrange(numClusters):
        clusterComp[i] = gacPathEntropy(IminuszW[np.ix_(initClusters[i], initClusters[i])])

    # compute initial(negative) affinity table(upper trianglar matrix), very slow
    if VERBOSE:
        print('   Computing initial table.')

    affinityTab = np.full(shape=(numClusters, numClusters), fill_value=np.inf)
    for j in xrange(numClusters):
        for i in xrange(j):
            affinityTab[i, j] = -1 * gacPathCondEntropy(IminuszW, initClusters[i], initClusters[j])

    affinityTab = (clusterComp + clusterComp.T) + affinityTab

    if VERBOSE:
        print('   Starting merging process')

    curGroupNum = numClusters
    while True:
        if np.mod(curGroupNum, 20) == 0 & VERBOSE:
            print('   Group count: ', str(curGroupNum))

        # Find two clusters with the best affinity
        minAff = np.min(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex2 = np.argmin(minAff)
        minIndex1 = minIndex1[minIndex2]
        if minIndex2 < minIndex1:
            minIndex1, minIndex2 = minIndex2, minIndex1

        # merge the two clusters

        new_cluster = np.unique(np.append(initClusters[minIndex1], initClusters[minIndex2]))

        # move the second cluster to be merged to the end of the cluster array
        # note that we only need to copy the end cluster's information to
        # the second cluster 's position
        if minIndex2 != curGroupNum:
            initClusters[minIndex2] = initClusters[-1]
            clusterComp[minIndex2] = clusterComp[curGroupNum - 1]
            # affinityTab is an upper triangular matrix
            affinityTab[: minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum - 1]
            affinityTab[minIndex2, minIndex2 + 1: curGroupNum - 1] = affinityTab[minIndex2 + 1:curGroupNum - 1,
                                                                     curGroupNum - 1]

        # update the first cluster and remove the second cluster
        initClusters[minIndex1] = new_cluster
        initClusters.pop()
        clusterComp[minIndex1] = gacPathEntropy(IminuszW[np.ix_(new_cluster, new_cluster)])
        clusterComp[curGroupNum - 1] = myInf
        affinityTab[:, curGroupNum - 1] = myInf
        affinityTab[curGroupNum - 1, :] = myInf
        curGroupNum = curGroupNum - 1
        if curGroupNum <= groupNumber:
            break

        # update the affinity table for the merged cluster
        for groupIndex1 in xrange(minIndex1):
            affinityTab[groupIndex1, minIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        for groupIndex1 in xrange(minIndex1 + 1, curGroupNum):
            affinityTab[minIndex1, groupIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1].reshape(-1) + clusterComp[minIndex1] + affinityTab[
                                                                                                            :minIndex1,
                                                                                                            minIndex1]
        affinityTab[minIndex1, minIndex1 + 1: curGroupNum] = clusterComp[minIndex1 + 1: curGroupNum].T + clusterComp[
            minIndex1] + affinityTab[minIndex1, minIndex1 + 1:curGroupNum]

    # generate sample labels
    clusterLabels = np.ones((numSample, 1))
    for i in xrange(len(initClusters)):
        clusterLabels[initClusters[i]] = i
    if VERBOSE:
        print('   Final group count: ', str(curGroupNum))

    return clusterLabels


def gacNNMerge(distance_matrix, NNIndex):
    # merge each vertex with its nearest neighbor
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
    #

    # NN indices
    sampleNum = distance_matrix.shape[0]
    clusterLabels = np.zeros((sampleNum, 1))
    counter = 1
    for i in xrange(sampleNum):
        idx = NNIndex[i, :2]
        assignedCluster = clusterLabels[idx]
        assignedCluster = np.unique(assignedCluster[np.where(assignedCluster > 0)])
        if len(assignedCluster) == 0:
            clusterLabels[idx] = counter
            counter = counter + 1
        elif len(assignedCluster) == 1:
            clusterLabels[idx] = assignedCluster
        else:
            clusterLabels[idx] = assignedCluster[0]
            for j in xrange(1, len(assignedCluster)):
                clusterLabels[np.where(clusterLabels == assignedCluster[j])] = assignedCluster[0]

    uniqueLabels = np.unique(clusterLabels)
    clusterNumber = len(uniqueLabels)

    initialClusters = []
    for i in xrange(clusterNumber):
        initialClusters.append(np.where(clusterLabels[:].flatten() == uniqueLabels[i])[0])

    return initialClusters


def gacBuildDigraph(distance_matrix, K, a):
    # Build directed graph
    # Input:
    #   - distance_matrix: pairwise distances, d_{i -> j}
    #   - K: the number of nearest neighbors for KNN graph
    #   - a: for covariance estimation
    #       sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
    #   - graphW: asymmetric weighted adjacency matrix,
    #               w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
    #	- NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    # NN indices
    N = distance_matrix.shape[0]
    # find 2*K NNs in the sense of given distances
    sortedDist = np.sort(distance_matrix, axis=1)
    NNIndex = np.argsort(distance_matrix, axis=1)
    NNIndex = NNIndex[:, :K + 1]

    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K + 1, 4)])) * a
    #########
    tmpNNDist = np.min(sortedDist[:, 1:], axis=1)
    while any(np.exp(- tmpNNDist / sig2) < 1e-5):  # check sig2 and magnify it if it is too small
        sig2 = 2 * sig2

    #########
    print('  sigma = ', str(np.sqrt(sig2)))

    # build graph
    ND = sortedDist[:, 1:K + 1]
    NI = NNIndex[:, 1:K + 2]
    XI = repmat(np.arange(0, N).reshape(-1, 1), 1, K)
    sig2 = np.double(sig2)
    ND = np.double(ND)
    graphW = sparse.csc_matrix((np.exp(-ND[:] * (1 / sig2)).flatten(), (XI[:].flatten(), NI[:].flatten())),
                               shape=(N, N)).todense()
    graphW += np.eye(N)

    return graphW, NNIndex


def gacCluster(distance_matrix, groupNumber, strDescr, K, a, z):
    # Graph Agglomerative Clustering toolbox
    # Input:
    #   - distance_matrix: pairwise distances, d_{i -> j}
    #   - groupNumber: the final number of clusters
    #   - strDescr: structural descriptor. The choice can be
    #                 - 'zeta':  zeta function based descriptor
    #                 - 'path':  path integral based descriptor
    #   - K: the number of nearest neighbors for KNN graph, default: 20
    #   - p: merging (p+1)-links in l-links algorithm, default: 1
    #   - a: for covariance estimation, default: 1
    #       sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
    #   - z: (I - z*P), default: 0.01
    # Output:
    #   - clusteredLabels: clustering results
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
    #
    # Please cite the following papers, if you find the code is helpful
    #
    # W. Zhang, D. Zhao, and X. Wang.
    # Agglomerative clustering via maximum incremental path integral.
    # Pattern Recognition, 46 (11): 3056-3065, 2013.
    #
    # W. Zhang, X. Wang, D. Zhao, and X. Tang.
    # Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
    # in Proceedings of European Conference on Computer Vision (ECCV), 2012.

    print('--------------- Graph Structural Agglomerative Clustering ---------------------');

    # initialization

    print('---------- Building graph and forming initial clusters with l-links ---------');
    [graphW, NNIndex] = gacBuildDigraph(distance_matrix, K, a);
    # from adjacency matrix to probability transition matrix
    graphW = np.array((1. / np.sum(graphW, axis=1))) * np.array(graphW)  # row sum is 1
    initialClusters = gacNNMerge(distance_matrix, NNIndex)

    print('-------------------------- Zeta merging --------------------------');
    clusteredLabels = gacMerging(graphW, initialClusters, groupNumber, strDescr, z);

    return clusteredLabels


def predict_ac_mpi(feat, nClass, nSamples, nfeatures):
    K = 20
    a = 1
    z = 0.01

    distance_matrix = cdist(feat, feat) ** 2
    # path intergral
    label_pre = gacCluster(distance_matrix, nClass, 'path', K, a, z)

    return label_pre[:, 0]


def bestMap(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)


def dataset_settings(dataset):
    if (dataset == 'MNIST-full') | (dataset == 'MNIST-test'):
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = ['same', 'same']
        test_batch_size = 100
    elif dataset == 'USPS':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = ['same', 'same']
        test_batch_size = 100
    elif dataset == 'FRGC':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = ['same', 'same']
        test_batch_size = 1231
    elif dataset == 'CMU-PIE':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = ['same', 'same']
        test_batch_size = 8
    elif dataset == 'YTF':
        kernel_sizes = [5, 4]
        strides = ['same', 'same']
        paddings = [2, 'same']
        test_batch_size = 100
    elif dataset == 'JD':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = ['same', 'same']
        test_batch_size = 100
    dropouts = [0.1, 0.1, 0.0]
    feature_map_sizes = [50, 50, 10]
    return kernel_sizes, strides, paddings, test_batch_size,dropouts,feature_map_sizes


def create_result_dirs(output_path, file_name):
    if not os.path.exists(output_path):
        print('creating log folder')
        os.makedirs(output_path)
        try:
            os.makedirs(os.path.join(output_path, '../params'))
        except:
            pass
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[1] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)


class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(output_path + "log.txt", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def kmeans(encoder_val_clean, y, nClusters, y_pred_prev=None, weight_initilization='k-means++', seed=42, n_init=40,
           max_iter=300):
    # weight_initilization = { 'kmeans-pca', 'kmean++', 'random', None }

    if weight_initilization == 'kmeans-pca':

        start_time = timeit.default_timer()
        pca = PCA(n_components=nClusters).fit(encoder_val_clean)
        kmeans_model = KMeans(init=pca.components_, n_clusters=nClusters, n_init=1, max_iter=300, random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

        end_time = timeit.default_timer()

    elif weight_initilization == 'k-means++':

        start_time = timeit.default_timer()
        kmeans_model = KMeans(init='k-means++', n_clusters=nClusters, n_init=n_init, max_iter=max_iter, n_jobs=15,
                              random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        D = 1.0 / euclidean_distances(encoder_val_clean, kmeans_model.cluster_centers_, squared=True)
        D **= 2.0 / (2 - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        centroids = kmeans_model.cluster_centers_.T
        centroids = centroids / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

        end_time = timeit.default_timer()
    if y[0] > 0:
        print('k-means: \t nmi =', normalized_mutual_info_score(y, y_pred), '\t arc =', adjusted_rand_score(y, y_pred),
              '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
              'K-means objective = {:.1f} '.format(kmeans_model.inertia_), '\t runtime =', end_time - start_time)

    if y_pred_prev is not None:
        print('Different Assignments: ', sum(y_pred == y_pred_prev), '\tbestMap: ', bestMap(y_pred, y_pred_prev),
              '\tdatapoints-bestMap*datapoints: ',
              encoder_val_clean.shape[0] - bestMap(y_pred, y_pred_prev) * encoder_val_clean.shape[0])

    return centroids, kmeans_model.inertia_, y_pred


def load_dataset(dataset_path):
    hf = h5py.File(dataset_path + '/data.h5', 'r')
    X = np.asarray(hf.get('data'), dtype='float32')
    X_train = (X - np.float32(127.5)) / np.float32(127.5)
    y_train = np.asarray(hf.get('labels'), dtype='int32')
    return X_train, y_train

def parse_function(filename,size):
    image_data = tf.read_file(filename)
    img = tf.image.decode_jpeg(image_data)
    img = tf.cast(img,tf.float32)
    img = tf.image.resize_images(img,[size,size])
    img = img / 127.5 - 1
    return img,-1

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    #slice超出部分会自动丢弃
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt


class Downsample(tf.keras.Model):

    def __init__(self, filters, size,strides,padding,dropout=0.5):
        super(Downsample, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=2,
                                            padding=padding,
                                            kernel_initializer=initializer,
                                            use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training):
        x = self.conv(x)
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(x,alpha=0.01)
        return x

class Encoder(tf.keras.Model):

    def __init__(self, feature_map_sizes,
                 dropouts, kernel_sizes, strides,
                 paddings):
        super(Encoder, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.bottom_layers = []
        self.first_layer = tf.keras.layers.Dropout(rate=dropouts[0])
        self.bottom_layers.append(self.first_layer)
        self.middle_layers_num = len(kernel_sizes)
        for i in range(self.middle_layers_num):
            l_ei = Downsample(feature_map_sizes[i],kernel_sizes[i],strides[i],padding=paddings[i],dropout=dropouts[i+1])
            self.bottom_layers.append(l_ei)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.last_layer = tf.keras.layers.Dense(units=feature_map_sizes[self.middle_layers_num],activation=keras.activations.tanh)
        #self.bottom_layers.append(self.last_layer)
    def call(self, x, training):
        # x shape == (bs, 256, 256, 3)
        outputs = []
        outputs.append(x)
        for i,layer in enumerate(self.bottom_layers):
            x = layer(x,training=training)
            if i > 0:
                outputs.append(x)
        x = self.flatten_layer(x)
        x = self.last_layer(x)
        outputs.append(x)
        return outputs

class Decoder(tf.keras.Model):

    def __init__(self,sec_shape, feature_map_sizes,
                 kernel_sizes, strides,
                 paddings):
        super(Decoder, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        self.top_layers = []
        self.first_layer = tf.keras.layers.Dense(units=sec_shape[1] * sec_shape[2] *sec_shape[3])
        #self.top_layers.append(self.first_layer)
        self.reshape_layer = keras.layers.Reshape(sec_shape[1:])

        self.middle_layers_num = len(kernel_sizes)
        for i in range(self.middle_layers_num):
            if i < self.middle_layers_num - 1:
                activation = keras.layers.LeakyReLU(alpha=0.01)
            else:
                activation = keras.activations.tanh
            l_di = keras.layers.Conv2DTranspose(filters=feature_map_sizes[-i-1],kernel_size=kernel_sizes[-i-1],strides=strides[-i-1],padding=paddings[-i-1],activation=activation)
            self.top_layers.append(l_di)
    def call(self, x):
        # x shape == (bs, 256, 256, 3)
        outputs = []
        x = self.first_layer(x)
        x = self.reshape_layer(x)
        outputs.append(x)
        for layer in self.top_layers:
            x = layer(x)
            outputs.append(x)
        return outputs

class AE(keras.models.Model):
    def __init__(self,encoder,decoder):
        super(AE, self).__init__(inputs=encoder.inputs,outputs=decoder.outputs[-1])
        self.encoder = encoder
        self.decoder = decoder
    def call(self, inputs, training=False):
        encoder_outs = self.encoder(inputs,training=training)
        decoder_outs = self.decoder(encoder_outs[-1])
        return encoder_outs,decoder_outs

def AE_loss(encoder_outs,decoder_outs):
    loss_recons = []
    for i in range(len(decoder_outs)):
        loss_recons.append(
            tf.losses.mean_squared_error(decoder_outs[i], decoder_outs[-i - 2])
        )
    loss_recon = sum(loss_recons)
    return loss_recon

def build_depict(input_var, feature_map_sizes=[50, 50],
                 dropouts=[0.1, 0.1, 0.1], kernel_sizes=[5, 5], strides=[2, 2],
                 paddings=[2, 2], hlayer_loss_param=0.1):
    # ENCODER
    input_layer = keras.layers.Input(tensor=input_var)
    encoder = Encoder(feature_map_sizes,
                     dropouts, kernel_sizes, strides,
                     paddings)
    encoder_outs = encoder(input_layer,training=True)
    #encoder_clean_outs = encoder(input_layer,training=False)
    # DECODER
    decoder_feature_map_sizes = [input_var.shape[-1].value] + feature_map_sizes[:-2]
    decoder = Decoder(encoder_outs[-2].shape, decoder_feature_map_sizes,
                 kernel_sizes, strides,
                 paddings)
    decoder_outs = decoder(encoder_outs[-1])
    # decoder_clean_outs = decoder(encoder_clean_outs[-1])
    ae = AE(encoder,decoder)
    #ae = keras.models.Model(inputs=input_layer,outputs=decoder.outputs[-1])
    # loss_recons = []
    # loss_clean_recons = []
    # for i in range(len(decoder_outs)):
    #     loss_recons.append(
    #         tf.losses.mean_squared_error(decoder_outs[i], encoder_clean_outs[-i-2])
    #     )
    #     loss_clean_recons.append(tf.losses.mean_squared_error(decoder_clean_outs[i], encoder_clean_outs[-i-2]))
    # loss_recon = sum(loss_recons)
    # loss_clean_recons = sum(loss_clean_recons)
    return ae #, loss_recon, loss_clean_recons


def train_depict_ae(dataset_name, dataset,dataset_val, ae,input_var, num_clusters, output_path,
                    batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, verbose=1, seed=42,
                    continue_training=False):
    encoder_outs, decoder_outs = ae(input_var, training=True)
    encoder_clean_outs, decoder_clean_outs = ae(input_var,training=False)
    loss_recon = AE_loss(encoder_clean_outs,decoder_outs)
    loss_clean_recon = AE_loss(encoder_clean_outs,decoder_clean_outs)
    weight_save_path = os.path.join(output_path, '../params/params_' + dataset_name + '_values_best.h5')
    best_val = np.inf
    last_update = 0
    keras.Model().predict()
    # Load if pretrained weights are available.
    if os.path.isfile(weight_save_path) & continue_training:
        ae.load_weights(weight_save_path)
        return
    else:
        # TRAIN MODEL
        if verbose > 1:
            encoder_clean = encoder_clean_outs[-1]
        iterator = dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        val_batch = dataset_val.make_one_shot_iterator().get_next()
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_recon)
        with K.get_session() as sess:
            sess.run(tf.global_variables_initializer())
            inputs_val = sess.run(val_batch)
            for epoch in range(num_epochs + 1):
                start = time.time()
                sess.run(iterator.initializer)
                num_batches = 0
                train_err = 0
                while True:
                    try:
                        inputs, labels = sess.run(next_batch)
                        # summery = sess.run([summery_op,gen_output],feed_dict={input_image_holder:input_image,target_placeholder:target})
                        train_err += sess.run([loss_recon,train_op],feed_dict={input_var:inputs})[0]
                        # if step % 10 == 0:
                        #     summery_writer.add_summary(summery, global_step=step)
                        num_batches += 1
                    except tf.errors.OutOfRangeError:
                        break
                valation_error = sess.run(loss_clean_recon,feed_dict={input_var:inputs_val})
                print("Epoch {} of {}".format(epoch + 1, num_epochs),
                      "\t  training loss:{:.6f}".format(train_err / num_batches),
                      "\t  valation loss:{:.6f}".format(valation_error),
                        "\t  time:{:.2f sec}".format(time.time()-start))
                # if epoch % 10 == 0:
                last_update += 1
                if valation_error < best_val:
                    last_update = 0
                    print("new best error: ", valation_error)
                    best_val = valation_error
                    ae.save_weights(weight_save_path)
                if last_update > 50:
                    break

                if (verbose > 1) & (epoch % 50 == 0) & labels[0] != -1:
                    # Extract MdA features
                    encoder_val_clean = []
                    y = []
                    while True:
                        try:
                            inputs, labels = sess.run(next_batch)
                            minibatch_x = sess.run(encoder_clean,feed_dict={input_var:inputs})
                            encoder_val_clean.append(minibatch_x)
                            y.append(labels)

                        except tf.errors.OutOfRangeError:
                            break
                    encoder_val_clean = np.concatenate(encoder_val_clean, axis=0)
                    y = np.concatenate(y, axis=0)
                    kmeans(encoder_val_clean, y, num_clusters, seed=seed)
        last_weight_path = os.path.join(output_path, '../params/params_' + dataset_name + '_values_last.h5')
        ae.save_weights(last_weight_path)
        #lasagne.layers.set_all_param_values(decoder, best_params_values)

def clustering(dataset_name,dataset,input_var, encoder, num_clusters, output_path, test_batch_size=100, seed=42,
               continue_training=False):
    encoder_clean = encoder(input_var,training=False)[-1]
    encoder_val_clean = []
    y = []
    iterator_training = dataset.make_one_shot_iterator()
    next_batch_training = iterator_training.get_next()
    dataset_train_size = 0
    with K.get_session() as sess:
        while True:
            try:
                inputs, labels = sess.run(next_batch_training)
                minibatch_x = sess.run(encoder_clean,feed_dict={input_var:inputs})
                encoder_val_clean.append(minibatch_x)
                y.append(labels)
            except tf.errors.OutOfRangeError:
                break
        encoder_val_clean = np.concatenate(encoder_val_clean, axis=0)
        y = np.concatenate(y, axis=0)
    # Extract MdA features
    # Check kmeans results
    kmeans(encoder_val_clean, y, num_clusters, seed=seed)
    initial_time = timeit.default_timer()
    if (dataset_name == 'MNIST-full') | (dataset_name == 'MNIST-test')| (dataset_name == 'FRGC') | (dataset_name == 'YTF') | (dataset_name == 'CMU-PIE') | (dataset_name == 'JD'):
        # K-means on MdA Features
        centroids, inertia, y_pred = kmeans(encoder_val_clean, y, num_clusters, seed=seed)
        y_pred = (np.array(y_pred)).reshape(np.array(y_pred).shape[0], )
        y_pred = y_pred - 1
    else:
        # AC-PIC on MdA Features
        if os.path.exists(os.path.join(output_path, '../params/pred' + dataset_name + '.pickle')) & continue_training:
            with open(os.path.join(output_path, '../params/pred' + dataset_name + '.pickle'), "rb") as input_file:
                y_pred = pickle.load(input_file, encoding='latin1')
        else:
            try:
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.addpath(eng.genpath('matlab'))
                targets_init = eng.predict_ac_mpi(
                    matlab.double(
                        encoder_val_clean.reshape(encoder_val_clean.shape[0] * encoder_val_clean.shape[1]).tolist()),
                    num_clusters, encoder_val_clean.shape[0], encoder_val_clean.shape[1])
                y_pred = (np.array(targets_init)).reshape(np.array(targets_init).shape[0], )
                eng.quit()
                y_pred = y_pred - 1
            except:
                y_pred = predict_ac_mpi(encoder_val_clean, num_clusters, encoder_val_clean.shape[0],
                                        encoder_val_clean.shape[1])
            with open(os.path.join(output_path, '../params/pred' + dataset_name + '.pickle'), "wb") as output_file:
                pickle.dump(y_pred, output_file)

        final_time = timeit.default_timer()
        print('AC-PIC: \t nmi =  ', normalized_mutual_info_score(y, y_pred),
              '\t arc = ', adjusted_rand_score(y, y_pred),
              '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
              '\t time taken = {:.4f}'.format(final_time - initial_time))
        centroids_acpic = np.zeros(shape=(num_clusters, encoder_val_clean.shape[1]))
        for i in range(num_clusters):
            centroids_acpic[i] = encoder_val_clean[y_pred == i].mean(axis=0)

        centroids = centroids_acpic.T
        centroids = centroids_acpic / np.sqrt(np.diag(np.matmul(centroids.T, centroids)))

    return np.int32(y_pred), np.float32(centroids)



def build_eml(n_out, W_initial=None):
    if W_initial is None:
        l_out = keras.layers.Dense(
            n_out,activation=keras.activations.softmax,bias_initializer='ones')
    else:
        l_out = keras.layers.Dense(
            n_out,activation=keras.activations.softmax,kernel_initializer=W_initial)
    return l_out
    #return keras.Model(inputs=encoder.inputs,outputs=l_out(encoder.outputs[-1]))

def train_depict(dataset_name, dataset_pred_train,dataset_pred_val, input_var, ae, num_clusters, y_pred_training,y_pred_val, output_path,
                 batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, prediction_status='soft',
                 rec_mult=1, clus_mult=1, centroids=None, init_flag=1, continue_training=False):
    ######################

    #   ADD RLC TO MdA   #
    ######################

    initial_time = timeit.default_timer()
    rec_lambda = rec_mult
    clus_lambda = clus_mult
    pred_normalizition_flag = 1
    num_batches = X.shape[0] // batch_size
    target_init = tf.placeholder(dtype=tf.int32,shape=[None])
    target_var = tf.placeholder(dtype=tf.float32,shape=[None,num_clusters])
    encoder_outs, decoder_outs = ae(input_var, training=True)
    encoder_clean_outs = ae.encoder(input_var, training=False)

    classifier = build_eml(n_out=num_clusters, W_initial=centroids)
    network_prediction_noisy = classifier(encoder_outs[-1],training=True)
    network_prediction_clean = classifier(encoder_clean_outs[-1], training=False)
    whole_model = keras.Model(inputs=ae.inputs,outputs=[ae.decoder.outputs[-1],classifier(ae.encoder.outputs[-1])])

    loss_clus_init = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(target_init,network_prediction_noisy))
    #params_init = lasagne.layers.get_all_params([decoder, network2], trainable=True)
    #`soft`目标是每个类都有一个概率，hard只有一个类的概率为1
    loss_clus = tf.reduce_mean(keras.losses.categorical_crossentropy(network_prediction_noisy,
                                                                target_var))
    loss_recons = AE_loss(encoder_clean_outs,decoder_outs)
    loss = clus_lambda * loss_recons + rec_lambda * loss_clus
    adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = adam.minimize(loss)
    params2 = lasagne.layers.get_all_params([decoder, network2], trainable=True)
    updates = lasagne.updates.adam(
        loss, params2, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var],
                               [loss, loss_recons, loss_clus], updates=updates)
    loss_init = clus_lambda * loss_recons + rec_lambda * loss_clus_init
    train_op_init = adam.minimize(loss_init)
    updates_init = lasagne.updates.adam(
        loss_init, params_init, learning_rate=learning_rate)
    train_fn_init = theano.function([input_var, target_init],
                                    [loss_init, loss_recons, loss_clus_init], updates=updates_init)

    test_fn = theano.function([input_var], network_prediction_clean)
    final_time = timeit.default_timer()
    weight_path = os.path.join(output_path, '../params/weights_' + dataset_name + '.h5')
    print("\n...Start DEPICT initialization")
    if init_flag:
        if os.path.exists(weight_path) & continue_training:
            whole_model.load_weights(weight_path)
        else:
            iter_train = dataset_pred_train.make_initializable_iterator()
            iter_val = dataset_pred_val.make_initializable_iterator()
            next_batch_val = iter_val.get_next()
            next_batch_training = iter_train.get_next()
            with K.get_session() as sess:
                sess.run(tf.variables_initializer(classifier.variables()))
                sess.run(iter_val.initalizer,feed_dict={"Pred_train:0":y_pred_val})
                inputs_val,inputs_labels_val,input_pred_val = sess.run(next_batch_val)
                for epoch in range(1000):
                    num_batches_train = 0
                    train_err, val_err = 0, 0
                    lossre_train, lossre_val = 0, 0
                    losspre_train, losspre_val = 0, 0
                    num_batches_train = 0
                    while True:
                        try:
                            inputs, y_labels,y_pred = sess.run(next_batch_training)
                            minibatch_error, lossrec, losspred = sess.run([loss_init, loss_recons, loss_clus_init, train_op], feed_dict={input_var: inputs})[:-1]

                            # if step % 10 == 0:
                            #     summery_writer.add_summary(summery, global_step=step)
                            train_err += minibatch_error
                            lossre_train += lossrec
                            losspre_train += losspred
                            num_batches_train += 1
                        except tf.errors.OutOfRangeError:
                            break
                    y_val_prob = sess.run(network_prediction_clean,feed_dict={input_var:inputs_val})
                    y_val_pred = np.argmax(y_val_prob, axis=1)

                    y_pred = np.zeros(X.shape[0])
                    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                        minibatch_inputs, targets, idx = batch
                        minibatch_prob = test_fn(minibatch_inputs)
                        minibatch_pred = np.argmax(minibatch_prob, axis=1)
                        y_pred[idx] = minibatch_pred
                    val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)

                    print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                          '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
                          '\t loss= {:.10f}'.format(train_err / num_batches_train),
                          '\t loss_reconstruction= {:.10f}'.format(lossre_train / num_batches_train),
                          '\t loss_prediction= {:.10f}'.format(losspre_train / num_batches_train),
                          '\t val nmi = {:.4f}  '.format(val_nmi))
                    last_update += 1
                    if val_nmi > best_val:
                        last_update = 0
                        print("new best val nmi: ", val_nmi)
                        best_val = val_nmi
                        best_params_values = lasagne.layers.get_all_param_values([decoder, network2])
                        # if (losspre_val / num_batches_val) < 0.2:
                        #     break

                    if last_update > 5:
                        break
            last_update = 0
            # Initilization
            y_targ_train = np.copy(y_pred_train)
            y_targ_val = np.copy(y_pred_val)
            y_val_prob = test_fn(X_val)
            y_val_pred = np.argmax(y_val_prob, axis=1)
            val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)
            best_val = val_nmi
            print('initial val nmi: ', val_nmi)
            best_params_values = lasagne.layers.get_all_param_values([decoder, network2])
            for epoch in range(1000):
                train_err, val_err = 0, 0
                lossre_train, lossre_val = 0, 0
                losspre_train, losspre_val = 0, 0
                num_batches_train = 0
                for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    minibatch_inputs, targets, idx = batch
                    minibatch_error, lossrec, losspred = train_fn_init(minibatch_inputs, np.int32(y_targ_train[idx]))
                    train_err += minibatch_error
                    lossre_train += lossrec
                    losspre_train += losspred
                    num_batches_train += 1

                y_val_prob = test_fn(X_val)
                y_val_pred = np.argmax(y_val_prob, axis=1)

                y_pred = np.zeros(X.shape[0])
                for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                    minibatch_inputs, targets, idx = batch
                    minibatch_prob = test_fn(minibatch_inputs)
                    minibatch_pred = np.argmax(minibatch_prob, axis=1)
                    y_pred[idx] = minibatch_pred

                val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)

                print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                      '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                      '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
                      '\t loss= {:.10f}'.format(train_err / num_batches_train),
                      '\t loss_reconstruction= {:.10f}'.format(lossre_train / num_batches_train),
                      '\t loss_prediction= {:.10f}'.format(losspre_train / num_batches_train),
                      '\t val nmi = {:.4f}  '.format(val_nmi))
                last_update += 1
                if val_nmi > best_val:
                    last_update = 0
                    print("new best val nmi: ", val_nmi)
                    best_val = val_nmi
                    best_params_values = lasagne.layers.get_all_param_values([decoder, network2])
                    # if (losspre_val / num_batches_val) < 0.2:
                    #     break

                if last_update > 5:
                    break

            lasagne.layers.set_all_param_values([decoder, network2], best_params_values)
            with open(os.path.join(output_path, '../params/weights' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(lasagne.layers.get_all_param_values([decoder, network2]), output_file)

    # Epoch 0
    print("\n...Start DEPICT training")
    y_prob = np.zeros((X.shape[0], num_clusters))
    y_prob_prev = np.zeros((X.shape[0], num_clusters))
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        y_prob[idx] = minibatch_prob

    y_prob_max = np.max(y_prob, axis=1)
    if pred_normalizition_flag:
        cluster_frequency = np.sum(y_prob, axis=0)
        y_prob = y_prob ** 2 / cluster_frequency
        y_prob = np.transpose(y_prob.T / np.sum(y_prob, axis=1))
    y_pred = np.argmax(y_prob, axis=1)

    print('epoch: 0', '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))
    if os.path.isfile(os.path.join(output_path, '../params/rlc' + dataset + '.pickle')) & continue_training:
        with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'),
                  "rb") as input_file:
            weights = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values([decoder, network2], weights)
    else:
        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            lossre = 0
            losspre = 0

            for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
                minibatch_inputs, targets, idx = batch

                # M_step
                if prediction_status == 'hard':
                    minibatch_err, lossrec, losspred = train_fn(minibatch_inputs,
                                                                np.ndarray.astype(y_pred[idx], 'int32'),
                                                                np.ndarray.astype(y_prob_max[idx],
                                                                                  'float32'))
                elif prediction_status == 'soft':
                    minibatch_err, lossrec, losspred = train_fn(minibatch_inputs,
                                                                np.ndarray.astype(y_prob[idx], 'float32'))

                minibatch_prob = test_fn(minibatch_inputs)
                y_prob[idx] = minibatch_prob
                train_err += minibatch_err
                lossre += lossrec
                losspre += losspred

            y_prob_max = np.max(y_prob, axis=1)
            if pred_normalizition_flag:
                cluster_frequency = np.sum(y_prob, axis=0)  # avoid unbalanced assignment
                y_prob = y_prob ** 2 / cluster_frequency
                # y_prob = y_prob / np.sqrt(cluster_frequency)
                y_prob = np.transpose(y_prob.T / np.sum(y_prob, axis=1))
            y_pred = np.argmax(y_prob, axis=1)

            # print('mse: ', mean_squared_error(y_prob, y_prob_prev))

            if mean_squared_error(y_prob, y_prob_prev) < 1e-7:
                with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'), "wb") as output_file:
                    pickle.dump(lasagne.layers.get_all_param_values([decoder, network2]), output_file)
                break
            y_prob_prev = np.copy(y_prob)

            print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                  '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                  '\t acc = {:.4f} '.format(bestMap(y, y_pred)), '\t loss= {:.10f}'.format(train_err / num_batches),
                  '\t loss_recons= {:.10f}'.format(lossre / num_batches),
                  '\t loss_pred= {:.10f}'.format(losspre / num_batches))

    # test
    y_pred = np.zeros(X.shape[0])
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        minibatch_inputs, targets, idx = batch
        minibatch_prob = test_fn(minibatch_inputs)
        minibatch_pred = np.argmax(minibatch_prob, axis=1)
        y_pred[idx] = minibatch_pred

    print('final: ', '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))
