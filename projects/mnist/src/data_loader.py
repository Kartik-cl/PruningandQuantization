import tensorflow as tf
from tensorflow.keras.datasets import mnist
from numpy import int64

'''
We load the data from raw files and do all the pre-processing here...
Exposed function is 'get_mnist_dataset' which returns tf.data.Dataset object.
'''

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
    return x_train,y_train,x_test,y_test

def _preprocess_data(x,y):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    x = x / 255.0
    y = y.astype(int64)
    return x,y

def _create_dataset(x,y,batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    return dataset

def get_mnist_dataset(batch_size=128):
    x_train, y_train, x_test, y_test = load_data()
    x_train, y_train = _preprocess_data(x_train,y_train)
    x_test,y_test = _preprocess_data(x_test,y_test)
    #dset_train = _create_dataset(x_train,y_train,batch_size)
    #dset_test = _create_dataset(x_test,y_test,batch_size)
    #print("Dataset created")
    #return dset_train,dset_test
    return x_train,y_train,x_test,y_test


