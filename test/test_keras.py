from keras.models import load_model
from keras.models import Model

model = load_model('models/lenet5_keras.h5')

# model.summary()
# for i in range(len(model.layers)):
#     print(model.layers[i].name)

# l = list(model.layers[0].input_shape)
# print(l)
model.summary()

length = len(model.layers)
x = model.layers[0].output
for i in range(1,length):
    x = model.layers[i](x)
# print(x.shape)
new_model = Model(inputs=model.layers[0].input, outputs=x)

weights = model.get_weights()
# new_model.get_weights()
new_model.set_weights(weights)
new_model.summary()

# import matplotlib.pyplot as plt

# a  = plt.imread("test_images/data1.png")

# a.resize(28,28)

# print(a)

# a.reshape(1,1,28,28)

# print(a)