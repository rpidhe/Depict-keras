import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import os
flag = tf.app.flags
img_height = 28
img_width = 28
img_size = img_height * img_width
output_size = 10

flag.DEFINE_integer("batch_size",100,"batch size")
flag.DEFINE_integer("iteration_step",1000,"iteration step")
FLAGS = flag.FLAGS
def full_connect(input_data,output_size,scope="linear",stddev=0.02, bias_start=0.0):
    input_size = input_data.shape[-1]
    with tf.variable_scope(scope):
        w = tf.get_variable("Matrix",[input_size,output_size],tf.float32,tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("Bias",[output_size],tf.float32,tf.constant_initializer(value=bias_start))
        return tf.matmul(input_data,w) + bias

if __name__ == "__main__":
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(FLAGS.batch_size).repeat()
    x_data = tf.placeholder(tf.float32,[None,img_height,img_width],"x_data")
    y_data = tf.placeholder(tf.int32,[None],"y_data")
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_tensor=x_data),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(output_size,activation=tf.nn.softmax)]
    )
    save_weight = "simple_weight.h5"
    if os.path.exists(save_weight):
        model.load_weights(save_weight)
    y_pred = model(x_data)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_data,y_pred)
    #loss = tf.keras.losses.categorical_crossentropy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1,output_type=tf.int32), y_data), dtype=tf.float32))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    iter = dataset.make_one_shot_iterator()
    next_data = iter.get_next()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.load_weights(save_weight)
        print('test accuracy %f' % accuracy.eval(feed_dict={x_data: x_test, y_data: y_test}))
        for i in range(FLAGS.iteration_step):
            x,y = session.run(next_data)
            feed_dict = {x_data: x, y_data: y}
            train_step.run(feed_dict=feed_dict)
            if i%50 == 0:
                train_accuracy = session.run(accuracy,feed_dict=feed_dict)
                print("Accuracy: %f" % train_accuracy)
        model.save_weights(save_weight,save_format='h5')
        print('test accuracy %f' % accuracy.eval(feed_dict={x_data: x_test, y_data: y_test}))