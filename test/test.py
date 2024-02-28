from keras.layers import Conv2D
from keras.models import Model
from keras.layers import Input, Dense,  GlobalAveragePooling2D
from keras.layers import MaxPooling2D, Activation,Lambda
from keras.utils import plot_model #需要安装graphviz 和pdyot
def keras_simple_model():
    inputs1 = Input((28, 28, 1))
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv1')(inputs1)
    x = Activation('relu')(x)
    x = Conv2D(4, (3, 3), activation=None, padding='same', name='conv2')(x)#layer_id=3替换的是这一层
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv3')(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), activation=None, padding='same', name='conv4')(x)


    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation=None)(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs1, outputs=x)
    return model
def replace_intermediate_layer_in_keras(model, layer_id, new_layer): #替代某层

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x) #根据层号寻找替换的层
        else:
            x = layers[i](x)

    new_model = Model(inputs=layers[0].input, outputs=x)
    return new_model

def insert_intermediate_layer_in_keras(model, layer_id, new_layer): #插入中间层

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        x = layers[i](x) #往后面接剩下的层

    new_model = Model(input=layers[0].input, output=x)
    return new_model

if __name__ == '__main__':
	model = keras_simple_model()
	model.summary()
	numOfParmeters=model.count_params()
	print("[INFO] The total number  of parmeters in model is {}".format(numOfParmeters))

	plotFileName="modelBeforeLayerReplacement.png"
	
	plot_model(model, to_file=plotFileName ,show_shapes=True)#模型图片会默认保存在py文件的路径下

	new_layer= Conv2D(4, (3, 3), activation=None, padding='same', name='conv2_repl', use_bias=False) #替代的层
	#new_layer= Conv2D(4, (3, 3), activation=None, padding='same', name='conv2_add', use_bias=False) #添加的层
	model = replace_intermediate_layer_in_keras(model, 3,new_layer)
	#model = insert_intermediate_layer_in_keras(model, 3,new_layer) 
	model.summary()
	numOfParmeters=model.count_params()
	print("[INFO] The total number  of parmeters in model is {}".format(numOfParmeters))
	plotFileName="modelAfterLayerReplacement1.png"	
	plot_model(model, to_file=plotFileName ,show_shapes=True)
