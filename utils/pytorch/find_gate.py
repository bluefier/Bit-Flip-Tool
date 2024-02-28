import numpy as np



def conv_output(model, layer, img):
    try:
        out_conv = layer.output
    except:
        raise Exception('Not layer named {}!'.format(layer))

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

def find_brelu_gate(x_test,model,input_shape=(1,32,32,3),layer_num=17):
    max_list = [-10000000] * layer_num
    for i in range(0, len(x_test)):
        print("epoch:" + str(i))
        for j in range(0, layer_num):
            data = x_test[i].reshape(input_shape)
            num = conv_output(model, acts[j], data)
            result = np.sort(num.flatten())
            maxnum = result[int(0.995 * len(num.flatten()))]
            max_in(max_list, maxnum, j)
    file = open("BNnet18_brelu_output_gate.txt", "a")
    for i in range(0, layer_num):
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