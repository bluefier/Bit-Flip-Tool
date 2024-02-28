import copy

import numpy as np
from anytree import Node, RenderTree, PreOrderIter
from build_tree import makeTree, is_not_basic_module, get_next_entity, get_relu_leaf_name, get_entity
import torch
import torch.nn as nn
from utils.keras.fault_inject import *
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd



def calculate_boundary(dataset, model,progress_callback):
    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model, root)
    pre_acts = pre_relu(model, root)
    if len(acts) == 0:
        return []
    print("需要修改的层数：{}".format(len(acts)))
    maxlist = gradient_optimiztion(dataset, model, acts, pre_acts,progress_callback)
    return maxlist


# 每个通道
def calculate_ch_boundary(dataset, model,progress_callback):


    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model,root)
    if len(acts) == 0:
        return []

    print("需要修改的层数：{}".format(len(acts)))
    return find_chrelu_gate(dataset, model, acts,progress_callback)


# 反向传播hook
def get_grad(grad_out):
    def back_hook(module, grad_input, grad_output):
        grad_out.append(grad_output)
    return back_hook


# 正向传播hook，返回矩阵
def get_output(outpt,inpt):
    def hook(module, input, output):
        outpt.append(output)
        inpt.append(input)
    return hook


# 获得最终的结果（包括错误注入）
def get_final_output(input, model):
    with torch.no_grad():
        out = model(input)
    return out


# 获得中间层输出，梯度
def conv_output(model, layer, input):
    mid_output = []
    grad=[]
    output=get_output(mid_output,grad)
    handle = layer.register_forward_hook(output)
    model(input)
    handle.remove()
    return mid_output,grad


# 返回relu的前一层
def pre_relu(model, root):
    acts = []
    leaves = root.leaves
    length = len(leaves)
    for i in range(length-1):
        node = leaves[i + 1]
        entity = get_entity(model, node.name)
        index = node.name.split('_')
        index = index[len(index) - 1]
        entity = getattr(entity, index)
        if type(entity) is nn.ReLU:
            pre_node = leaves[i]
            acts.append(pre_node.name)
    return acts


class BRelu(nn.Module):

    def __init__(self, boundary):
        super(BRelu, self).__init__()
        self.boundary = boundary

    def forward(self, x):
        x = torch.clamp(x, 0, self.boundary)
        return x


# 损失函数，传入正常输出值和偏差输出值
def loss_fn(out1, out2):
    # 找到最大值和第二大的值
    _, index = torch.max(out1, 1)
    loss = nn.CrossEntropyLoss()
    res = loss(out2, index)
    print(f'the loss is {res}')
    return res

def rectify_res(b_list,idx,out):
    rate=0.005
    max_val=0
    for i in range(len(out)):
        out_flat = out[i].flatten()
        k = int((1-rate) * len(out_flat))
        high_values = torch.topk(out_flat, k, largest=False)
        max_val = max(high_values.values.max(),max_val)
    ret=(b_list[idx[0] + 2] + b_list[idx[1] + 2] + b_list[idx[2] + 2]) / 3
    res=rate*ret+max_val
    return res.item()

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re


# 传入参数，分别对应输入模型，权重路径，relu前一层，输入数据,迭代次数，运行设备，错误注入率，boundarie
def gradient_optimiztion(inputs, model, acts, pre_acts,callback):
    total = len(inputs)
    count = 0
    boundarys = [-100000 for i in range(len(acts))]
    for input in inputs:
        count += 1
        new_model = model
        progress = int(100 * count / total)
        callback(progress)
        input=torch.tensor(input).permute(0,3,1,2)
        for i in range(len(acts)):
            print(f'i is {i}--------------------')
            b_list = []
            fixed_layer=pre_acts[i]
            epoch = 50
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            b = 0.1  # 初始化最初b值
            rate = 0.01  # 比特反转率
            model = model.to(device)
            input=torch.tensor(input,dtype=torch.float32).to(device)
            fixed_layer=fixed_layer.replace('_','.')
            weight = fixed_layer + '.weight'
            lr = 1e-2       #boundary增加率

            index = acts[i].split('_')
            index = index[len(index) - 1]
            entity = get_entity(model, acts[i])

            # relu前一层
            p_index = pre_acts[i].split('_')
            p_index = p_index[len(p_index) - 1]
            fix_entity = getattr(entity, index)

            # 得到梯度
            layer_output ,grad= conv_output(model, fix_entity,input)
            grad = grad[0][0]
            # 对导数求其倒数
            reci_grad = 1 / grad
            reci_grad[reci_grad > 20000000] = 0

            # print('the reciprocal_grad', reci_grad)
            zp=layer_output[0].max()
            normal_out = get_final_output(input, model)
            loss = []
            for j in range(epoch):
                if j == 0:
                    new_model = copy.deepcopy(model)
                    state_dict = new_model.state_dict()
                    weight = state_dict[weight]
                    num_elements = weight.numel()
                    random_index = torch.randint(0, num_elements, ((int)(num_elements * rate),))
                    r_index = np.unravel_index(np.asarray(random_index), weight.shape)
                    weight[r_index] *= 1e6
                b_list.append(b)
                print(f'epoch is {j}-----------------------------')
                print(f'b:{b}---------------------------------------')
                #relu层
                b_entity=get_entity(new_model, acts[i])

                setattr(b_entity, index, BRelu(b))
                # 得到偏差输出
                inject_out = get_final_output(input, new_model)

                minus_out = inject_out - normal_out
                minus_out = minus_out.sum()


                # print("max_out", layer_output[0].max())
                boundary = reci_grad * minus_out + layer_output[0]
                boundary[boundary < 0] = 0
                b += grad.max().item() * lr
                loss.append(np.array(loss_fn(normal_out, inject_out).to('cpu')))
            #平滑损失
            y_av = moving_average(loss, 5)
            ave_loss = torch.tensor(y_av).flatten()
            _, idx = torch.topk(ave_loss,3, largest=False)
            b_value=rectify_res(b_list,idx,layer_output)  #idx从2开始索引，取最小loss的三个b值
            if b_value>boundarys[i]:
                boundarys[i]=b_value
    return boundarys

def find_chrelu_gate(dataset, model, acts,progress_callback):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    change_num = len(acts)
    max_list = [[-10000000 for i in range(5000)] for j in range(change_num)]

    total = len(dataset)
    count = 0
    # assuming dataset is a DataLoader or a list of PyTorch tensors
    for i, data in enumerate(dataset):

        count += 1
        progress = int(100 * count / total)
        progress_callback(progress)


        print("epoch:", i)
        data = data.permute(0, 3, 1, 2).to(device)
        data = data.to(device)
        for j in range(change_num):
            out = conv_output(model, acts[j], data)
            for item in out:
                # max_list[j][0] =out.shape[-1]
                if item.shape[1]>5000:
                    return False
                #批组数
                for m in range(item.shape[0]):
                    #通道数
                    for n in range(item.shape[1]):
                        #卷积层
                        if len(item.shape)==4:
                            out_flat=torch.topk(item[m,n,:,:].flatten(),int(0.995*len(item[m,n,:,:].flatten())),largest=False)
                            max_num=out_flat.values.max()
                            max_list[j][n]=max(max_num,max_list[j][n])
                        #全连接层
                        elif  len(item.shape)==2:
                            max_list[j][n]=max(max_list[j][n],item[m][n])
                        else:
                            return False

    # tensor,
    length = []
    mean = []

    df = pd.DataFrame()
    for i in range(len(max_list)):
        count = 0
        total = 0
        value = []
        for item in max_list[i]:
            if isinstance(item, torch.Tensor):
                total += item.item()
                value.append(item.item())
                count += 1
            else:
                value.append(item)
        length.append(count)
        mean.append(total/count)
        df[f'layer_{i}'] = value
    print(length)
    print(mean)

    # # 将DataFrame写入Excel文件
    df.to_excel('data/output.xlsx', index=False)

    return max_list
