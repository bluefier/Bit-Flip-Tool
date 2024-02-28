import numpy as np
import keras

def calculate_boundary(dataset,model):
    acts = []
    layers_num = len(model.layers)+1
    for i in range(layers_num-1):
        if "relu" in model.layers[i].name:
            acts.append(model.get_layer(model.layers[i-1].name))

    length = len(acts)
    print("需要修改的层数：{}".format(length))
    return find_brelu_gate(dataset,model,acts)

def calculate_ch_boundary(dataset,model):
    acts = []
    layers_num = len(model.layers)+1
    for i in range(layers_num-1):
        if "relu" in model.layers[i].name:
            acts.append(model.get_layer(model.layers[i-1].name))

    return find_chrelu_gate(dataset,model,acts)

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

# 两种找gate都要找线性层的gate，就相当于将线性层看做很厚的1x1的特征图
def find_brelu_gate(x_test,model,acts):
    # 原理：100张图片，输入一张，计算所有层的最大上界，然后输入下一张，再次计算上界，如果计算出来比已经存在的上界大就更新上界。
    change_num = len(acts)
    max_list = [-10000000] * change_num
    l = len(x_test)
    for i in range(0, l):
        print("epoch:" + str(i))
        for j in range(0, change_num):
            data = x_test[i]
            print("打印图片形状：" + str(x_test[i].shape))
            num = conv_output(model, acts[j], data)
            result = np.sort(num.flatten())
            maxnum = result[int(0.995 * len(num.flatten()))]
            max_in(max_list, maxnum, j)
            
    return max_list
    
def find_chrelu_gate(x_test,model,acts):
    # 原理：对每一层，用1张图片计算出该层特征图，然后把每个通道中所有参数的最大值，保存到max_list[j][n]中。同样是更大就更新。
    # 本质上来说和上面的是一样的。只是for的先后不同而已。
    # act:所有relu的前一层。
    length = len(acts)
    # max_list = [[-10000000] * 501] * length # 不能这样写的原因是python会将列表当做对象，所以加进去的只是引用，修改会连带修改。
    max_list = [[-10000000 for j in range(501)] for i in range(length)] # 501代表一个特征图最多有500个通道
    print(length)
    l = len(x_test)
    for j in range(0, length):
        print("layer:" + str(j + 1))
        for i in range(0, l):
            data = x_test[i] # （batch,宽高，通道）
            num = conv_output(model, acts[j], data)
            print(num.shape)
            max_list[j][0] = num.shape[-1] # 重复写入，这里可以用if优化一下
            # print(max_list[0][0])
            # print(max_list[1][0])
            if num.shape[-1]>500:
                return False
            if len(num.shape)==4:
                # 说明是卷积层
                for m in range(0,num.shape[0]): # m是batch，就是1。（不知道为什么要这样写）
                    for n in range(0,num.shape[-1]): # n是通道
                        result = np.sort(num[m,:,:,n].flatten()) # 第j层的计算出的特征图的第n个通道拉直，按小到大排序。
                        # max_list[index + n] = max(result[int(0.97 * len(result))],max_list[index + n])
                        max_list[j][n+1] = max(result[int(0.97 * len(result))],max_list[j][n+1])
                
            elif len(num.shape)==2:
                # 线性层
                for m in range(0,num.shape[0]): # m是batch，就是1。（不知道为什么要这样写）
                    for n in range(0,num.shape[-1]): # n是通道
                        result = num[m][n] # 第j层的计算出的特征图的第n个通道拉直，按小到大排序。
                        max_list[j][n+1] = max(result,max_list[j][n+1])
                
            else:
                print("出错了")

    return max_list