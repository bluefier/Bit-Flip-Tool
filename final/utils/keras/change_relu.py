import tensorflow.compat.v1 as tf
from keras import backend as K
import os
import keras
from keras.layers import Activation,Lambda,ReLU
from keras.models import Model
import numpy as np

# pytorch修改

# tf.disable_eager_execution()
# config = tf.compat.v1.ConfigProto()
# sess = tf.compat.v1.Session(config = config)
# K.set_session(sess)

# @tf.function
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
        
        val = []
        for i in range(0,x.shape[-1]):
            val.append( max_value[i]) # val[i]就是该层第i个卷积核对应relu的上界。

        x = tf.clip_by_value(x,0,val) # 映射x
    if alpha != 0.0:
        alpha = _to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x
# def create_chrelu(values):
#     def chrelu(x):
#         return relu(x,max_value = values)
#     return chrelu

def chrelu(x,values):
    return relu(x,max_value=values)

def fix(model,boundarys:list):
    length = len(model.layers)
    acts = []
    for i in range(length):
        if "relu" in model.layers[i].name:
            acts.append(model.layers[i].name)  # type: ignore
    return replace_intermediate_layer_in_keras(model,acts,boundarys)
    
def fix_ch(model,boundarys:list):
    length = len(model.layers)
    acts = []
    for i in range(length):
        if "relu" in model.layers[i].name:
            acts.append(model.layers[i].name)  # type: ignore
    return replace_intermediate_layer_in_keras(model,acts,boundarys)
    
def replace_intermediate_layer_in_keras(model, acts, boundarys): #替代某层
    layers = [m for m in model.layers]
    x = layers[0].output
    j=0
    length = len(layers)
    length2 = len(acts)
    # print(length)
    new_model = "出错了"
    if len(np.array(boundarys).shape)==2:
        print("说明是通道置换")
        for i in range(1, length):
            if j<length2 and layers[i].name == acts[j]:
                l = boundarys[j][0]
                # print(l)
                # print(boundarys[j][1:l+1])
                # x = Activation(create_chrelu(boundarys[j][1:l+1]),name="ChRelu"+str(j))(x) # 自定义的问题。
                x = chrelu(x,boundarys[j][1:l+1])
                j+=1
            else:
                x = layers[i](x)
            # print(x.shape)
        new_model = Model(inputs=layers[0].input, outputs=x)
    else:
        print("说明是层置换")
        for i in range(1, length):
            if j<length2 and layers[i].name == acts[j]:
                # 根据层名修改
                x = ReLU(max_value=boundarys[j],name="Brelu"+str(i))(x)
                j+=1
            else:
                x = layers[i](x)
            # print(x.shape)
        new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model # 由于只修改了

def insert_intermediate_layer_in_keras(model, layer_id, new_layer): #插入中间层
    layers = [l for l in model.layers]
    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x) #往后面接剩下的层

    new_model = Model(input=layers[0].input, output=x)
    return new_model

# # 用这个需要自定义层，不能保存这个max_value到模型文件中。
# def create_relu(max_value):
#     def ownrelu(x):
#         return K.relu(x,max_value = max_value)
#     return ownrelu



def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    Args:
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    Returns:
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)
def _constant_to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    This is slightly faster than the _to_tensor function, at the cost of
    handling fewer cases.

    Args:
        x: An object to be converted (numpy arrays, floats, ints and lists of
          them).
        dtype: The destination type.

    Returns:
        A tensor.
    """
    return tf.constant(x, dtype=dtype)



# Argument `initial_value` (Tensor("activation/zeros:0", shape=(64,), dtype=float32))
#  could not be lifted out of a `tf.function`. (Tried to create variable with name='None').
#  To avoid this error, when constructing `tf.Variable`s inside of `tf.function` 
# you can create the `initial_value` tensor in a `tf.init_scope` or pass a callable `initial_value` 
# (e.g., `tf.Variable(lambda : tf.truncated_normal([10, 40]))`).
#  Please file a feature request if this restriction inconveniences you.

# Call arguments received by layer "activation" (type Activation):
#   • inputs=tf.Tensor(shape=(None, 8, 8, 64), dtype=float32)
