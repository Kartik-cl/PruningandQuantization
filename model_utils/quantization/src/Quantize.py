import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import sys
from tensorflow_model_optimization.sparsity import keras as sparsity
sys.path.append('C:/Users/Sapna/Desktop/PruningAndQuantization/model_utils/pruning/src')
from Prune import Prune
'''
qwt : quantization aware training

0.1234 (32 bits in memory) == 0.0110011000101
qunatization - lowering the precision of our numbers
0.0110011000101 == (qunatize with 8 bits 1 for sign... and rest 7 for fractional parts) 
0.0110011 == 0.1225

qwt quantization aware training -  retraining - 8bit simulated env ... getting back the accuracy.

'''
class Quantize():
    def __init__(self,class_obj):
        self.class_obj = class_obj
        self.parent_map = {}
        self.prune_obj = None

    def get_layer_name(self,name):
        name = name.split('/')[0]
        name = name.replace(':0','')
        name = name.replace('prune_low_magnitude_','')
        if name[-2]=='_':
           name = name[:-2]
        return name
    '''
    def create_qwt_graph(self):
        self.sess = tf.keras.backend.get_session()
        tf.contrib.quantize.create_training_graph(self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        #self.class_obj.compile_model()
    '''
    def quantize(self,x):
        #return tf.quantization.fake_quant_with_min_max_vars(x,tf.reduce_min(x),tf.math.maximum(tf.reduce_max(x),tf.constant(0.0)))
        return tf.quantization.quantize_and_dequantize(x,tf.reduce_min(x),tf.math.maximum(tf.reduce_max(x),tf.constant(0.0)))
        
    def create_qwt_graph(self,shape_):
        input_layer = Input(shape=shape_, name='input_layer')
        map_layer_name_obj={} 
        input_layers = []
        #self.class_obj.model.summary()
        
        for layer in self.class_obj.model.layers:
            if('input' in layer.name):
                continue

            print('layer name',layer.name)
            inputs_to_layer = []
            if(isinstance(layer.input,list) ):
                for inp_layer in layer.input:
                    if('input' in inp_layer.name):
                        tmp = self.quantize(inp_layer)
                        map_layer_name_obj[self.get_layer_name(inp_layer.name)]=tmp
                        input_layers.append(inp_layer)
                        #print(inp_layer.name,'found')
                    inputs_to_layer.append(map_layer_name_obj[self.get_layer_name(inp_layer.name)])
                x = layer(inputs_to_layer)
            else:
                if('input' in layer.input.name):
                    tmp =self.quantize(layer.input)
                    map_layer_name_obj[self.get_layer_name(layer.input.name)]=tmp
                    input_layers.append(layer.input)
                x = layer(map_layer_name_obj[self.get_layer_name(layer.input.name)])
            layer_name = layer.name
            del layer
            x._name = layer_name
            x = self.quantize(x) #
            map_layer_name_obj[self.get_layer_name(layer_name)] = x

            
        del self.class_obj.model
        self.class_obj.model = Model(inputs=input_layers,outputs=x)
        
    def train_qwt_graph(self,train_args,to_prune):
        if(to_prune):
            prune_callbacks = [
                        sparsity.UpdatePruningStep(),
                        #sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
                    ]
            train_args['call_backs'] = train_args['call_backs']+prune_callbacks
        else:
            train_args['call_backs'] = train_args['call_backs']
        self.class_obj.train_model(**train_args)

    def eval_qwt_graph(self,eval_args):
        return self.class_obj.eval_model(**eval_args)
    
    def remove_quant_layers(self):
        input_layers = []
        map_layer_name_obj={}
        parent_node = None
        for layer in self.class_obj.model.layers:
            if('input' in layer.name):
                continue
            elif('tf_op_layer' in layer.name):
                parent_node = layer.input
                del layer
                continue
            else:
                if(isinstance(layer.input,list)):
                    inputs_to_layer = []
                    for inp_layer in layer.input:
                        if('input' in inp_layer.name):
                            map_layer_name_obj[self.get_layer_name(inp_layer.name)]=inp_layer
                            input_layers.append(inp_layer)
                        if('tf_op_layer' in inp_layer.name):
                            map_layer_name_obj[self.get_layer_name(inp_layer.name)]=parent_node
                        inputs_to_layer.append(map_layer_name_obj[self.get_layer_name(inp_layer.name)])
                    x = layer(inputs_to_layer)
                else:
                    if('input' in layer.input.name):
                        map_layer_name_obj[self.get_layer_name(layer.input.name)]=layer.input
                        input_layers.append(layer.input)
                    if('tf_op_layer' in layer.input.name):
                        map_layer_name_obj[self.get_layer_name(layer.input.name)]=parent_node
                    x = layer(map_layer_name_obj[self.get_layer_name(layer.input.name)])
                    print(layer.name,x.name)
                layer_name=layer.name
                del layer
                #x._name = layer_name
                map_layer_name_obj[self.get_layer_name(layer_name)] = x
                print(x.name)
                
        del self.class_obj.model
        print(input_layers)
        self.class_obj.model = Model(inputs=input_layers,outputs=x)
        #self.class_obj.model = self.prune_obj.strip_extra_params(self.class_obj.model)
        self.class_obj.compile_model()

    def save_qwt_graph(self,path):
        #g = tf.get_default_graph()
        #tf.contrib.quantize.create_eval_graph(input_graph=g)
        self.class_obj.save_model(os.path.join(path,'quantized_model.h5'))

    def create_pruned_and_qwt_graph(self,shape_,sparsity_factor):
        self.prune_obj = Prune()
        self.class_obj.model = self.prune_obj.get_prunable_model(self.class_obj.model,shape_,sparsity_factor,remove_1=False)
        self.create_qwt_graph(shape_)




