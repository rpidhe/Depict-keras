import os

import argparse

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from functions import *
import socket
import time
from tensorflow.keras.datasets import mnist
if __name__ == "__main__":
    ############################## settings ##############################
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--dataset', default='MNIST-test')
    parser.add_argument('--continue_training', action='store_true', default=False)
    parser.add_argument('--datasets_path', default='/datasets/')
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', default=4000)
    parser.add_argument('--reconstruct_hyperparam', default=1.)
    parser.add_argument('--cluster_hyperparam', default=1.)
    parser.add_argument("--num_clusters",type=int,default=3)
    parser.add_argument('--architecture_visualization_flag', default=1)
    parser.add_argument('--loss_acc_plt_flag', default=1)
    parser.add_argument('--verbose', default=2)
    args = parser.parse_args()

    ############################## Logging ##############################
    output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + args.dataset \
                 # + time.strftime("%d-%m-%Y_") + \
                 # time.strftime("%H:%M:%S") + '_' + args.dataset + '_' + socket.gethostname()
    pyscript_name = os.path.basename(__file__)
    create_result_dirs(output_path, pyscript_name)
    sys.stdout = Logger(output_path)
    print(args)
    print('----------')
    print(sys.argv)

    # fixed random seeds
    seed = args.seed
    np.random.seed(args.seed)
    rng = np.random.RandomState(seed)
    #theano_rng = MRG_RandomStreams(seed)
    #lasagne.random.set_rng(np.random.RandomState(seed))
    learning_rate = args.learning_rate
    dataset_name = args.dataset
    datasets_path = args.datasets_path
    #dropouts = args.dropouts
    #feature_map_sizes = args.feature_map_sizes
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    cluster_hyperparam = args.cluster_hyperparam
    reconstruct_hyperparam = args.reconstruct_hyperparam
    verbose = args.verbose

    ############################## Load Data  ##############################
    if dataset_name == "JD":
        num_samples = len(os.listdir(datasets_path))
        input_size = 64
        dimensions = [input_size,input_size,3]
        dataset = tf.data.Dataset.list_files(os.path.join(datasets_path, "*.jpg"), shuffle=True).shuffle(1000).map(
            lambda x :parse_function(x,input_size)).batch(batch_size).repeat()
        num_clusters = args.n_clusters
    else:
        (_,_),(X,y) = mnist.load_data()
        X = np.expand_dims(X,axis=-1)
        dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(buffer_size=12000).batch(batch_size).repeat()
        num_clusters = len(np.unique(y))
        num_samples = len(y)
        dimensions = list(X.shape[1:])
    print('dataset: %s \tnum_samples: %d \tnum_clusters: %d'
          % (dataset_name, num_samples, num_clusters))

    #input_var = T.tensor4('inputs')
    kernel_sizes, strides, paddings, test_batch_size, dropouts,feature_map_sizes = dataset_settings(dataset_name)
    feature_map_sizes[-1] = num_clusters
    print(
        '\n... build DEPICT model...\nfeature_map_sizes: %s \tdropouts: %s \tkernel_sizes: %s \tstrides: %s \tpaddings: %s'
        % (str(feature_map_sizes), str(dropouts), str(kernel_sizes), str(strides), str(paddings)))
    ######## ######################  Build DEPICT Model  ##############################
    ae = build_depict(dimensions,feature_map_sizes=feature_map_sizes,
                                                                    dropouts=dropouts, kernel_sizes=kernel_sizes,
                                                                    strides=strides,
                                                                    paddings=paddings)

    ############################## Pre-train DEPICT Model   ##############################
    print("\n...Start AutoEncoder training...")
    initial_time = timeit.default_timer()
    train_depict_ae(dataset_name, dataset, dimensions, num_clusters, output_path,
                    batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                    verbose=verbose, seed=seed, continue_training=args.continue_training)

    ############################## Clustering Pre-trained DEPICT Features  ##############################
    y_pred, centroids = clustering(dataset, X, y, input_var, encoder, num_clusters, output_path,
                                   test_batch_size=test_batch_size, seed=seed, continue_training=args.continue_training)

    ############################## Train DEPICT Model  ##############################
    train_depict(dataset, X, y, input_var, decoder, encoder, loss_recons, num_clusters, y_pred, output_path,
                 batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs,
                 learning_rate=learning_rate, rec_mult=reconstruct_hyperparam, clus_mult=cluster_hyperparam,
                 centroids=centroids, continue_training=args.continue_training)

    final_time = timeit.default_timer()

    print('Total time for ' + dataset + ' was: ' + str((final_time - initial_time)))
