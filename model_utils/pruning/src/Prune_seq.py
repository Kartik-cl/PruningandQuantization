import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
import os
class Prune():
    '''
    def get_prunable_model(self,model,shape_,sparsity_factor):
        prunable_model = Model()
        prunable_model.add(Input(shape=shape_, name='input'))
        pruning_params = {
            'pruning_schedule': sparsity.ConstantSparsity(target_sparsity=sparsity_factor,
                                                   begin_step=0,frequency=1)  
                            }
        for layer in model.layers:
            if('input' in layer.name):
                continue
            if('conv' in layer.name or 'fc' in layer.name):
                prunable_model.add(sparsity.prune_low_magnitude(
                                            layer,
                                            **pruning_params
                                            )
                                )
            else:
                prunable_model.add(layer)
        return prunable_model
    '''
    
    def get_layer_name(self,name):
        name = name.split('/')[0]
        name = name.replace(':0','')
        #name = name.replace('_1','')
        return name

    def get_prunable_model(self,model,shape_,sparsity_factor):
        prunable_model = []
        input_layer = Input(shape=shape_, name='input_layer')
        pruning_params = {
            'pruning_schedule': sparsity.ConstantSparsity(target_sparsity=sparsity_factor,
                                                   begin_step=0,frequency=1)  
                            }
        #x = input_layer
        dict_layers_input={}
        dict_layers_input[self.get_layer_name(model.layers[1].input.name)]=input_layer
        for layer in model.layers:
            print(dict_layers_input.keys())
            if('input' in layer.name):
                #print(layer.name)
                continue
            if('conv' in layer.name or 'fc' in layer.name):
                pruned_layer = sparsity.prune_low_magnitude(
                                            layer,
                                            **pruning_params
                                            )
                #print(layer.input.name , layer.name)
                inputs_to_layer = []
                if(layer.input is list):
                    for inp_layer in layer.input:
                        inputs_to_layer.append(dict_layers_input[self.get_layer_name(inp_layer)])
                    x = pruned_layer(inputs_to_layer)
                else:
                    x = pruned_layer(dict_layers_input[self.get_layer_name(layer.input.name)])
                dict_layers_input[layer.name] = x 
            else:
                inputs_to_layer = []
                if(layer.input is list):
                    for inp_layer in layer.input:
                        inputs_to_layer.append(dict_layers_input[self.get_layer_name(inp_layer)])
                    x = layer(inputs_to_layer)
                else:
                    x = layer(dict_layers_input[self.get_layer_name(layer.input.name)])
                dict_layers_input[layer.name] = x 
            print(dict_layers_input.keys())
        return Model(inputs=input_layer,outputs=x)

    def print_pruning_stats(self,model):
        for i, w in enumerate(model.get_weights()):
            print(
                "{} -- Total:{}, Zeros: {:.2f}%".format(
                    model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
                )
            )

    def strip_extra_params(self,model):
        return sparsity.strip_pruning(model)

    def train_with_pruning(self,class_obj,train_args,test_args,input_shape,path,accuracy_before_pruning=0.9,init_sparsity=0.5,target_sparsity=0.9,acc_threshold=0.05,max_epochs=5):
        curr_sparsity = init_sparsity
        is_model_learning = True
        
        while(curr_sparsity<target_sparsity and is_model_learning):
            print(curr_sparsity)
            curr_acc=0
            class_obj.model = self.get_prunable_model(class_obj.model,input_shape,curr_sparsity)
            class_obj.compile_model()
            curr_epoch = 1
            while((accuracy_before_pruning-curr_acc)>acc_threshold and curr_epoch<=max_epochs):
                logdir = '../logs/'
                callbacks = [
                    sparsity.UpdatePruningStep(),
                    sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
                ]
                train_args['call_backs'] = callbacks
                class_obj.train_model(**train_args)
                curr_acc = class_obj.eval_model(**test_args)
                curr_epoch += 1
            
            if(curr_epoch>max_epochs):
                is_model_learning=False

            
            class_obj.model = self.strip_extra_params(class_obj.model)
            class_obj.save_model(os.path.join(path,'model_sparse_'+str(curr_sparsity*100)+'.h5'))
            self.print_pruning_stats(class_obj.model)
            curr_sparsity = curr_sparsity + (1-curr_sparsity)/2
