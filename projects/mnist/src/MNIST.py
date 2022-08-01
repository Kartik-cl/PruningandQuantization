#source : https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/01-TF2.0-Overview
import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import models

'''
This is the class for the model structure.
It has following basic functionalities :
    create_model
    load_model
    save_model
    test
    train
    predict 
'''

class MNIST():
    def __init__(self):
        self.num_classes = 10
        self.model = None
        self.input = Input(shape=(28, 28, 1), name='input')
        self.step_size = 128


    def create_model(self):
        x = Conv2D(24, kernel_size=(6, 6), strides=1,activation='relu',name='conv1')(self.input)
        x = BatchNormalization(scale=False, beta_initializer=Constant(0.01),name='bn1')(x)
        x = Activation('relu',name='act1')(x)
        
        x = Conv2D(48, kernel_size=(5, 5), strides=2,name='conv2')(x)
        x = BatchNormalization(scale=False, beta_initializer=Constant(0.01),name='bn2')(x)
        x = Activation('relu',name='act2')(x)
        
        x = Conv2D(64, kernel_size=(4, 4), strides=2,name='conv3')(x)
        x = BatchNormalization(scale=False, beta_initializer=Constant(0.01),name='bn3')(x)
        x = Activation('relu',name='act3')(x)
        
        x = Flatten()(x)
        x = Dense(200,name='fc1')(x)
        x = BatchNormalization(scale=False, beta_initializer=Constant(0.01))(x)
        x = Activation('relu',name='act4')(x)
        predications = Dense(self.num_classes, activation='relu', name='fc2')(x)
        self.model = Model(inputs=[self.input], outputs=predications)
        print("Model created")
        
    def predict(self, x):
        logits = self.model(x)
        return tf.argmax(logits,axis=1)

    
    def compile_model(self):
        self.model.compile(
            #loss=tf.keras.losses.categorical_crossentropy,
            loss = 'sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

    def train_model(self,x_train,y_train,x_test,y_test,batch_size,epochs,call_backs=[]):
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=call_backs,
          validation_data=(x_test, y_test))

    def eval_model(self,x,y,batch_size):
        loss,acc = self.model.evaluate(x, y, batch_size=batch_size)
        return acc


    def save_model(self,path):
        self.model.save(path)

    def restore_model(self,path):
        self.model = models.load_model(path)
