from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.sparsity import keras as sparsity
from MNIST import MNIST
from data_loader import get_mnist_dataset,load_data,_preprocess_data
from tensorflow.keras.models import Model
import sys
sys.path.append('C:/Users/Sapna/Desktop/PruningAndQuantization/model_utils/quantization/src')
from Quantize import Quantize

net = MNIST()
net.create_model()

### To be read from config
batch_size = 128
### To be added to MNIST class
input_shape = (28,28,1) 

x_train,y_train,x_test,y_test = load_data()
x_train,y_train = _preprocess_data(x_train,y_train)
x_test,y_test = _preprocess_data(x_test,y_test)

x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:200]
y_test = y_test[:200]

# prune_callbacks = [
#                     sparsity.UpdatePruningStep(),
#                 ]

train_args = {
    'x_train':x_train,
    'y_train':y_train,
    'x_test':x_test,
    'y_test':y_test,
    'batch_size':128,
    'epochs':2,
     'call_backs':[]
}

eval_args = {
    'x':x_test,
    'y':y_test,
    'batch_size':batch_size
}
save_folder_path = '../model/quantized_models/'


'''
LOAD THE WEIGHTS HERE - PRUNED MODEL WEIGHTS 
'''
net.restore_model('../model/mnist.h5')
net.compile_model()
res = net.eval_model(**eval_args)
print('Normal model acc',res)
print(net.model.predict(x_test[0].reshape(1,28,28,1)))
quantize = Quantize(net)

quantize.create_qwt_graph((28,28,1))
net.compile_model()
quantize.train_qwt_graph(train_args,False)
quantize.remove_quant_layers()
net.compile_model()
res = quantize.eval_qwt_graph(eval_args)
print('quantized model acc',res)
print(quantize.class_obj.model.predict(x_test[0].reshape(1,28,28,1)))
#print(net.model.summary())

quantize.save_qwt_graph(save_folder_path)

# sess = tf.keras.backend.get_session()
# print(sess.run(net.model.layers[1].kernel).min())
# print(sess.run(net.model.layers[1].kernel).max())
