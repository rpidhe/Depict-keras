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
    parser.add_argument('--dataset_name', default='MNIST-test')
    parser.add_argument('--continue_training', action='store_true', default=True)
    parser.add_argument('--datasets_path', default= 'F:/Project/Commodity and Logo Recognization/AI_JD/crop_dataset/pair_show_new/')
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', default=4000)
    parser.add_argument('--reconstruct_hyperparam', default=1.)
    parser.add_argument('--cluster_hyperparam', default=1.)
    parser.add_argument("--n_clusters",type=int,default=3)
    parser.add_argument('--architecture_visualization_flag', default=1)
    parser.add_argument('--loss_acc_plt_flag', default=1)
    parser.add_argument('--verbose', default=1)
    args = parser.parse_args()

    ############################## Logging ##############################
    output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + args.dataset_name
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
    learning_rate = args.learning_rate
    dataset_name = args.dataset_name
    datasets_path = args.datasets_path
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    cluster_hyperparam = args.cluster_hyperparam
    reconstruct_hyperparam = args.reconstruct_hyperparam
    verbose = args.verbose

    ############################## Load Data  ##############################
    if dataset_name == "JD":
        images_path = os.listdir(datasets_path)
        num_samples = len(images_path)
        val_size = 1000
        X_train, X_val = train_test_split(
            images_path, test_size=val_size, random_state=42)
        input_size = 64
        dimensions = [input_size,input_size,3]
        y = np.array([-1]*num_samples)
        dataset_full = tf.data.Dataset.from_tensor_slices(images_path).map(
            lambda x: parse_function(x, input_size)).batch(batch_size)
        num_clusters = args.n_clusters
    elif dataset_name == "MNIST-test":
        (_,_),(X,y) = mnist.load_data()
        #np.save("mnist-test-X.npy",X)
        #np.save("mnist-test-Y.npy",y)
        X = (X - np.float32(127.5)) / np.float32(127.5)
        X = np.expand_dims(X, axis=-1)
        num_samples = X.shape[0]
        val_size = 1000
        dataset_full = tf.data.Dataset.from_tensor_slices(X)
        num_clusters = len(np.unique(y))
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
    input_var = tf.placeholder(dtype=tf.float32,shape=[None] + dimensions)
    ae = build_depict(input_var,feature_map_sizes=feature_map_sizes,
                                                                    dropouts=dropouts, kernel_sizes=kernel_sizes,
                                                                    strides=strides,
                                                                    paddings=paddings)
    ############################## Pre-train DEPICT Model   ##############################
    print("\n...Start AutoEncoder training...")
    initial_time = timeit.default_timer()
    ae_best_weight_save_path = train_depict_ae(dataset_name, dataset_full,y,ae, input_var, num_clusters, output_path,val_size,
                    batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                    verbose=verbose, seed=seed, continue_training=args.continue_training)
    ############################## Clustering Pre-trained DEPICT Features  ##############################
    y_pred, centroids = clustering(dataset_name,dataset_full,y,input_var, ae, num_clusters,ae_best_weight_save_path, output_path,
                                   test_batch_size=test_batch_size, seed=seed, continue_training=args.continue_training)

    ############################## Train DEPICT Model  ##############################
    train_depict(dataset_name, dataset_full,y,y_pred, input_var,val_size, ae, num_clusters, ae_best_weight_save_path,output_path,
                 batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs,
                 learning_rate=learning_rate, rec_mult=reconstruct_hyperparam, clus_mult=cluster_hyperparam,
                 centroids=centroids, continue_training=args.continue_training)

    final_time = timeit.default_timer()

    print('Total time for ' + dataset_name + ' was: ' + str((final_time - initial_time)))
