import tensorflow as tf
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from tensorflow.contrib.factorization import KMeansClustering
num_clusters = 10

(_,_),(x,y) = tf.keras.datasets.mnist.load_data()
x = x.reshape(len(x),-1)
num_epochs = 100
def input_fn(num_epochs):
    return tf.train.limit_epochs(tf.convert_to_tensor(x,tf.float32),num_epochs=num_epochs)
def eval_input_fn(num_epochs):
    return tf.train.limit_epochs((tf.convert_to_tensor(x,tf.float32),tf.convert_to_tensor(y,tf.int32)),num_epochs=num_epochs)

kmeans = KMeansClustering(num_clusters,
initial_clusters=KMeansClustering.KMEANS_PLUS_PLUS_INIT,use_mini_batch=False,kmeans_plus_plus_num_retries=10
                                                   )
kmeans.train(lambda :input_fn(num_epochs),max_steps=200)
predict = kmeans.predict_cluster_index(input_fn(1))
print(list(predict))
print(normalized_mutual_info_score(y,predict))