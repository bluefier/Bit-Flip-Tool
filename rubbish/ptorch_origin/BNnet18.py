# -*- coding: UTF-8 -*- 


from __future__ import print_function
import keras
import random
import copy
import math
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow.compat.v1 as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from fault_inject import inject_SBF
from fault_inject import inject_layer_MBF
 
from keras.layers import Activation,Lambda
from keras.utils import np_utils
from keras.models import load_model
from fault_inject import inject_SBF
from fault_inject import inject_layer_MBF
from fault_inject import ConvertToFloatHex
import csv
from keras.layers import merge
from keras import optimizers,regularizers
import numpy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)
K.set_session(sess)

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
weight_decay = 1e-4

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def BNnet18(img_input):
    x = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    #conv_block1
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    #conv_block2
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    #conv_block3
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    #conv_block4
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    
    x = AveragePooling2D(pool_size=4, strides = (1,1))(x1)
    x = Flatten()(x)
    x = Dense(10,activation='softmax',kernel_initializer='he_normal')(x)
    
    return x


def create_relu(max_value):
    def ownrelu(x):
        return K.relu(x,max_value = max_value)
    return ownrelu
'''
gates = open("BNnet18_max_output_gate.txt", "r")
lines = gates.readlines()
gates.close()
'''
'''
gates = open("BNnet18_brelu_output_gate.txt", "r")
lines = gates.readlines()
gates.close()
'''
def CoarsnessBNnet18(img_input):
    index = 0
    x = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(img_input)
    x = BatchNormalization()(x)
    x = Activation(create_relu(max_value = float(lines[index])))(x)
    index+=1
    
    #conv_block1
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    #conv_block2
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    #conv_block3
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    #conv_block4
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    x = Activation(create_relu(max_value = float(lines[index])))(x1)
    index+=1
    
    x = AveragePooling2D(pool_size=4, strides = (1,1))(x)
    x = Flatten()(x)
    x = Dense(10,activation='softmax',kernel_initializer='he_normal')(x)
    
    return x


def _constant_to_tensor(x, dtype):
    return tf.constant(x, dtype=dtype)

def relu(x, alpha=0.0, max_value=None, threshold=0.0):
    dtype = getattr(x, "dtype", K.floatx())
    if alpha != 0.0:
        if max_value is None and threshold == 0:
            return tf.nn.leaky_relu(x, alpha=alpha)

        if threshold != 0:
            negative_part = tf.nn.relu(-x + threshold)
        else:
            negative_part = tf.nn.relu(-x)

    clip_max = max_value is not None

    if threshold != 0:
        x = x * tf.cast(tf.greater(x, threshold), dtype=dtype)
    elif max_value == 6:
        x = tf.nn.relu6(x)
        clip_max = False
    else:
        x = tf.nn.relu(x)

    if clip_max:
        
        zero = tf.convert_to_tensor(tf.zeros(x.shape[-1]),dtype = x.dtype.base_dtype)
        cmpval = tf.Variable(zero)
        sess.run(tf.global_variables_initializer())
        val = sess.run(cmpval)
        for i in range(0,x.shape[-1]):
            val[i] = max_value[i] # val[i]就是该层第i个卷积核对应relu的上界。
        value = tf.convert_to_tensor(tf.assign(cmpval,val)) # 利用广播机制得到上界和下界0
        x = tf.clip_by_value(x,zero,value) # 映射x
    if alpha != 0.0:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def create_chrelu(values):
    def chrelu(x):
        return relu(x,max_value = values)
    return chrelu

gates = open("BNnet18_chrelu_output_gate1.txt", "r")
strlines = gates.readlines()
gates.close()
lines = []
for i in strlines:
  lines.append(float(i))


def ChreluBNnet18(img_input):
    start = 0
    end = 0
    x = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(img_input)
    end += x.shape[-1]
    x = BatchNormalization()(x)
    x = Activation(create_chrelu(lines[start:end]))(x)
    start += x.shape[-1]
    
    #conv_block1
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    end += x1.shape[-1]
    x1 = BatchNormalization()(x1)
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 64,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    #conv_block2
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 128,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    #conv_block3
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 256,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    #conv_block4
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (2,2), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    end += x1.shape[-1]
    x1 = Activation(create_chrelu(lines[start:end]))(x1)
    start += x1.shape[-1]
    
    x1 = Conv2D(filters = 512,kernel_size = (3,3), strides = (1,1), padding = 'same',kernel_initializer='he_normal',kernel_regularizer = regularizers.l2(weight_decay))(x1)
    x1 = BatchNormalization()(x1)
    #x1 = Activation('relu')(x1)
    
    end += x1.shape[-1]
    x = Activation(create_chrelu(lines[start:end]))(x1)
    start += x.shape[-1]
    
    x = AveragePooling2D(pool_size=4, strides = (1,1))(x)
    x = Flatten()(x)
    x = Dense(10,activation='softmax',kernel_initializer='he_normal')(x)
    
    return x



img_input = Input(shape=(32,32,3))
#model = Model(inputs=img_input,output=BNnet18(img_input))  
#model = Model(inputs=img_input,output=CoarsnessBNnet18(img_input))
model = Model(inputs=img_input,output=ChreluBNnet18(img_input))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
'''
model_type = 'BNNet18v1'
      
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    model.save_weights('bnnet18_cifar10.h5')
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)
    model.save_weights('BNnet18_cifar10.h5')
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])    
'''

model.load_weights("saved_models/cifar10_BNNet18v1_model.169.h5", by_name=True)
#loss,accuracy = model.evaluate(x_test,y_test)
#print(loss,accuracy)    

def conv_output(model, layer_name, img):

    try:
        out_conv = layer_name.output
    except:
        raise Exception('Not layer named {}!'.format(layer_name))

    conv_visualization_model = keras.Model(inputs=model.input, outputs=out_conv)

    return conv_visualization_model.predict(img)
    
def max_in(l, num, i):
    x = l[i]
    if num > x:
        l[i] = num
    else: 
        return

acts = []          
for i in range(1,18):
  acts.append(model.get_layer("batch_normalization_" + str(i)))  

def find_max_gate():
    max_list = [-10000000] * 17
    for i in range(0, 100):
        print("epoch:" + str(i))
        for j in range(0, 17):
            data = x_test[i].reshape(1,32,32,3)
            num = conv_output(model, acts[j], data)
            maxnum = np.max(num.flatten())
            max_in(max_list, maxnum, j)
    file = open("BNnet18_max_output_gate.txt", "a")
    for i in range(0, 17):
        file.write(str(max_list[i])+'\n')
    file.close()  
    
def find_brelu_gate():
    max_list = [-10000000] * 17
    for i in range(0, 100):
        print("epoch:" + str(i))
        for j in range(0, 17):
            data = x_test[i].reshape(1,32,32,3)
            num = conv_output(model, acts[j], data)
            result = np.sort(num.flatten())
            maxnum = result[int(0.995 * len(num.flatten()))]
            max_in(max_list, maxnum, j)
    file = open("BNnet18_brelu_output_gate.txt", "a")
    for i in range(0, 17):
        file.write(str(max_list[i])+'\n')
    file.close() 
    
def find_chrelu_gate():
    max_list = [-10000000] * (3904)
    file = open("BNnet18_chrelu_output_gate1.txt", "a")
    index = 0
    for j in range(0, 17):
        print("layer:" + str(j + 1))
        for i in range(0, 100):
            data = x_test[i].reshape(1,32,32,3)
            num = conv_output(model, acts[j], data)
            for m in range(0,num.shape[0]):
                for n in range(0,num.shape[3]):
                    result = np.sort(num[m,:,:,n].flatten()) # 第j层的第n个卷积核的输出，按小到大排序。
                    max_list[index + n] = max(result[int(0.97 * len(result))],max_list[index + n])
        index += num.shape[3]
    for i in range(0,3904):              
        file.write(str(max_list[i]) + '\n') 
    file.close()

convs = [] 
for i in range(1, 18):   
    convs.append(model.get_layer("conv2d_" + str(i)))
  
def find_weights():
  input_tensors = [
     model.inputs[0],
     model.sample_weights[0],     
     model.targets[0],     
     K.learning_phase(), ]
  index_list = []   
  for i in convs:     
    grads = K.gradients(model.total_loss,i.trainable_weights)     
    total_gradient = []     
    get_gradients = K.function(inputs=input_tensors,outputs=grads)     
    for j in range(0, 100):             
        gradient = get_gradients([x_test[100*j: 100*j + 100],        
        np.ones(x_test[100*j: 100*j + 100].shape[0]),        
        y_test[100*j: 100*j + 100], 
        0])          
        total_gradient += gradient     
    a = np.array(total_gradient[0])     
    index = np.unravel_index(a.argmax(), a.shape)     
    index_list.append(index) 
  file = open("BNnet18_weights.txt", "a")
  for i in range(0,len(index_list)):              
        file.write(str(index_list[i]) + '\n') 
  file.close()

#find_max_gate()
#find_brelu_gate()
#find_chrelu_gate()
#find_weights()
'''
with open("resnet18_weights.txt", "r") as f:
  ff=f.readlines()
  print(type(eval(ff[0])))
'''
'''
for i in  range(0,10):
    loss,accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)
'''

'''


bit_num = 2
for LayerNum in range(0,20):
    model.load_weights("resnet18_cifar10.h5", by_name=True)
    LayerName = convs[LayerNum]
    weight,bias = LayerName.get_weights()
    faulty_weight = weight[index_list[LayerNum][0],index_list[LayerNum][1],index_list[LayerNum][2],index_list[LayerNum][3]]
    weight[index_list[LayerNum][0],index_list[LayerNum][1],index_list[LayerNum][2],index_list[LayerNum][3]] = inject_SBF(faulty_weight,bit_num)
    weights = (weight,bias)
    LayerName.set_weights(weights)
    score = model.evaluate(x_test, y_test)
    print(score)
    print(score[1])
'''   
'''
data = x_test[0].reshape(1,32,32,3)
#val = conv_output(model, convs[1], data)
#val = conv_output(model, model.get_layer('batch_normalization_2'), data)
val = conv_output(model, model.get_layer('activation_1'), data)
file = open("outputACT.txt", "a")   
#file = open("outputBN2.txt", "a")         
#file = open("outputCONV2.txt", "a")    
for m in range(0,val.shape[0]):
  for n in range(0,val.shape[3]):
    for i in range(0,val.shape[1]):
      for j in range(0,val.shape[2]):
        file.write(str(val[m,i,j,n])+'\n')
file.close()
'''


bit_num = 2
LayerNum = 0
LayerName = convs[0]
weight,bias = LayerName.get_weights()

'''
file = open("noBNweight0.txt", "a")    
for m in range(0,weight.shape[3]):
  for n in range(0,weight.shape[2]):
    for i in range(0,weight.shape[0]):
      for j in range(0,weight.shape[1]):
        file.write(str(weight[i,j,n,m])+'\n')
file.close()


file = open("noBNbias0.txt", "a")    
for m in range(0,bias.shape[0]):
   file.write(str(bias[m])+'\n')
file.close()
'''

#index_list = [(0,2,0,48)]
#index_list = [(1,0,1,25)]
#index_list = [(0,2,2,42)]
index_list = [(0,2,2,14)]
print(weight.shape)
faulty_weight = weight[index_list[LayerNum][0],index_list[LayerNum][1],index_list[LayerNum][2],index_list[LayerNum][3]]
print(faulty_weight)
weight[index_list[LayerNum][0],index_list[LayerNum][1],index_list[LayerNum][2],index_list[LayerNum][3]] = inject_SBF(faulty_weight,bit_num)
print(weight[index_list[LayerNum][0],index_list[LayerNum][1],index_list[LayerNum][2],index_list[LayerNum][3]])
weights = (weight,bias)
LayerName.set_weights(weights)
#score = model.evaluate(x_test, y_test)
#print(score)
#print(score[1])

data = x_test[0].reshape(1,32,32,3)
'''
file = open("input0.txt", "a")    
for m in range(0,data.shape[0]):
  for n in range(0,data.shape[3]):
    for i in range(0,data.shape[1]):
      for j in range(0,data.shape[2]):
        file.write(str(data[m,i,j,n])+'\n')
file.close()
'''

#val = conv_output(model, convs[1], data)
val = conv_output(model, model.get_layer('activation_1'), data)
#val = conv_output(model, model.get_layer('batch_normalization_2'), data)
#file = open("outputBN2fch_14.txt", "a")         
file = open("outputact1fch_14.txt", "a")    
#file = open("outputconv2fch_14.txt", "a")   
for m in range(0,val.shape[0]):
  for n in range(0,val.shape[3]):
    for i in range(0,val.shape[1]):
      for j in range(0,val.shape[2]):
        file.write(str(val[m,i,j,n])+'\n')
file.close()

'''
f = open("vgg.csv","a")
for i in range(0,200):
  data = []
  sum = 0
  ErrorRate = i / 10000;
  for j in range(0,15):
    model.load_weights('LeNet_5.h5', by_name=True)
    for LayerNum in range(0,5):
      LayerName = convs[LayerNum]
      weight,bias = LayerName.get_weights()
      faulty_weight = inject_layer_MBF(weight,ErrorRate)
      weights = (faulty_weight,bias)
      LayerName.set_weights(weights)
    score = model.evaluate(x_test, y_test)
    sum += score[1]
    data.append(score[1]);
  average = sum/5
  data.append(average);
  writer = csv.writer(f)
  writer.writerow(data)
f.close()   
'''