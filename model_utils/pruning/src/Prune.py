import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import os
class Prune():
    
    def get_layer_name(self,name,remove_1):
        name = name.split('/')[0]
        name = name.replace(':0','')
        if remove_1 and name[-2]=='_':
            name = name[:-2]
        return name

    def get_prunable_model(self,model,shape_,sparsity_factor,remove_1=False,output_key='output'):
        prunable_model = []
        input_layer = Input(shape=shape_, name='input_layer')
        pruning_params = {
            'pruning_schedule': sparsity.ConstantSparsity(target_sparsity=sparsity_factor,
                                                   begin_step=0,frequency=1)  
                            }

        '''
        The map_layer_name_obj dictionary will contain the name of layer and corresponding layer object.
        Lets say the model was  a=conv2d b=BN, c=FC and d=concat
            inp
             | 
             a
            / \
           b   c
           \  /
            d
        
        and after pruning... pruning defined by ' (dash)
            inp
             | 
             a'
            / \
           b   c'
           \  /
            d           
        so dictionary becomes
        {
            a:pruned_conv obj
            b:BN obj
            c:pruned_FC obj
            d:concat obj
        }

        now, we parse the layers of original model to create pruned model
        general parsing is done by x = Layer(input)

        so here we do by 
        x = Layer(map_layer_name_obj[input.name]) ...if there are multiple inputs... we sent a list of layer objects
        and then we add this layer object x to curr layer name
        map_layer_name_obj[currlayer.name] =x 
        '''
        map_layer_name_obj={} 
        
        #print('model inputs')
        input_layers = []
        input_layers_name=[]
        #for inputs in model.inputs:
        #    print('    ',self.get_layer_name(inputs.name,remove_1))        
        #    map_layer_name_obj[self.get_layer_name(inputs.name,remove_1)]=inputs
        #    input_layers.append(inputs)
        #print(input_layers)
        remove_1=False
        #print(map_layer_name_obj.keys())
        output_layers = []
        for layer in model.layers:
            #print(layer.name)
            if('input' in layer.name):
                continue
            
            elif('conv' in layer.name or 'fc' in layer.name):
                pruned_layer = sparsity.prune_low_magnitude(
                                            layer,
                                            **pruning_params
                                            )
                inputs_to_layer = []
                if(isinstance(layer.input,list)):
                    for inp_layer in layer.input:
                        if('input' in inp_layer.name and inp_layer.name not in input_layers_name):
                            map_layer_name_obj[self.get_layer_name(inp_layer.name,remove_1)]=inp_layer
                            input_layers.append(inp_layer)
                            input_layers_name.append(inp_layer.name)
                            #print(layer.input.name,'found')
                        inputs_to_layer.append(map_layer_name_obj[self.get_layer_name(inp_layer,remove_1)])
                    x = pruned_layer(inputs_to_layer)
                else:
                    #print(self.get_layer_name(layer.input.name,remove_1))
                    if('input' in layer.input.name and layer.input.name not in input_layers_name):
                        map_layer_name_obj[self.get_layer_name(layer.input.name,remove_1)]=layer.input
                        input_layers.append(layer.input)
                        input_layers_name.append(layer.input.name)
                        print(layer.input.name,'found')
                    x = pruned_layer(map_layer_name_obj[self.get_layer_name(layer.input.name,remove_1)])
                map_layer_name_obj[layer.name] = x 
                x._name=layer.name
            else:
                inputs_to_layer = []
                if(isinstance(layer.input,list) ):
                    for inp_layer in layer.input:
                        if('input' in inp_layer.name and inp_layer.name not in input_layers_name):
                            map_layer_name_obj[self.get_layer_name(inp_layer.name,remove_1)]=inp_layer
                            input_layers.append(inp_layer)
                            input_layers_name.append(inp_layer.name)
                            #print(inp_layer.name,'found')
                        inputs_to_layer.append(map_layer_name_obj[self.get_layer_name(inp_layer.name,remove_1)])
                    x = layer(inputs_to_layer)
                else:
                    #print(self.get_layer_name(layer.input.name,remove_1))
                    if('input' in layer.input.name and layer.input.name not in input_layers_name):
                        map_layer_name_obj[self.get_layer_name(layer.input.name,remove_1)]=layer.input
                        input_layers.append(layer.input)
                        input_layers_name.append(layer.input.name)
                        #print(layer.input.name,'found')
                    x = layer(map_layer_name_obj[self.get_layer_name(layer.input.name,remove_1)])
                x._name=layer.name
                map_layer_name_obj[layer.name] = x 
            if(output_key in layer.name):
                output_layers.append(x)    
        if(len(output_layers)==0):
            output_layers.append(x)
        return Model(inputs=input_layers,outputs=output_layers)

    def print_pruning_stats(self,model):
        for i, w in enumerate(model.get_weights()):
            print(
                "{} -- Total:{}, Zeros: {:.2f}%".format(
                    model.weights[i].name, w.size, np.sum(w == 0) / w.size * 100
                )
            )

    def strip_extra_params(self,model):
        return sparsity.strip_pruning(model)

    def train_with_pruning(self,class_obj,train_args,test_args,input_shape,path,accuracy_before_pruning=0.58,init_sparsity=0.5,target_sparsity=0.9,acc_threshold=0.02,max_epochs=30,output_key='output'):
        curr_sparsity = init_sparsity
        is_model_learning = True
        remove_1=False
        while(curr_sparsity<target_sparsity and is_model_learning):
            print('Current sparsity : ',curr_sparsity)
            curr_acc=0
            #class_obj.model.summary()
            class_obj.model = self.get_prunable_model(class_obj.model,input_shape,curr_sparsity,remove_1,output_key)
            remove_1=False
            #self.print_pruning_stats(class_obj.model)
            class_obj.compile_model()
            class_obj.model.summary()
            curr_epoch = 0
            while((accuracy_before_pruning-curr_acc)>acc_threshold and curr_epoch<max_epochs):
                logdir = '../logs/'
                prune_callbacks = [
                    sparsity.UpdatePruningStep(),
                    #sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
                ]
                train_args['call_backs'] = train_args['call_backs']+prune_callbacks
                class_obj.train_model(**train_args)
                curr_acc = class_obj.eval_model(**test_args)
                curr_epoch += 1
                print('Current Epoch : ',curr_epoch, 'curr_acc :', curr_acc)
            
            if(((accuracy_before_pruning-curr_acc)>acc_threshold) and  curr_epoch>=max_epochs):
                print('The model is not learning. Stopping the pruning process')
                is_model_learning=False
                #class_obj.save_model(os.path.join(path,'prunable_model_epoch_'+str(curr_epoch)+'_sparse_'+str(curr_sparsity*100)+'_valacc_'+str(curr_acc)+'.h5'))            
                class_obj.model = self.strip_extra_params(class_obj.model)
                class_obj.save_model(os.path.join(path,'model_epoch_'+str(curr_epoch)+'_sparse_'+str(curr_sparsity*100)+'_valacc_'+str(curr_acc)+'.h5'))
            
                return 

            #class_obj.save_model(os.path.join(path,'prunable_model_epoch_'+str(curr_epoch)+'_sparse_'+str(curr_sparsity*100)+'_valacc_'+str(curr_acc)+'.h5'))             
            class_obj.model = self.strip_extra_params(class_obj.model)
            class_obj.save_model(os.path.join(path,'model_epoch_'+str(curr_epoch)+'_sparse_'+str(curr_sparsity*100)+'_valacc_'+str(curr_acc)+'.h5'))
            self.print_pruning_stats(class_obj.model)
            curr_sparsity = curr_sparsity + (1-curr_sparsity)/2

