import torch
from anytree import Node, RenderTree, PreOrderIter
from torchvision import models
import torch.nn as nn
from build_tree import makeTree,is_not_basic_module,get_next_entity,get_relu_leaf_name,get_entity,get_leaf_name
from utils.pytorch.change_relu1 import BRelu



def find_nonzero_indices(tensor):
    nonzero_indices = torch.nonzero(tensor)
    return nonzero_indices

def count_different_weights(model1_path, model2_path):
    # 加载整个模型
    model1 = torch.load(model1_path, map_location='cpu')
    model2 = torch.load(model2_path, map_location='cpu')

    # 提取模型参数字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    different_params = []
    num_different_params = 0
    for key in state_dict1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            different_params.append(key)
            dif_array = state_dict1[key] - state_dict2[key]
            nonzero_indices = find_nonzero_indices(dif_array)

            print("Nonzero indices:")
            print(nonzero_indices)
            num_different_params += len(nonzero_indices)

    return num_different_params, different_params


def find_brelu_modules(module, brelu_modules):

    for child_module in module.children():


        # print(type(child_module),isinstance(child_module,BRelu))
        if isinstance(child_module, BRelu):
            brelu_modules.append(child_module)
            print(child_module.boundary)
        elif isinstance(child_module, nn.Module):
            find_brelu_modules(child_module, brelu_modules)



if __name__ == "__main__":
    model1_path = "C:/Users/ChenPanda/Desktop/fixed_inject_models.pth"
    model2_path = "C:/Users/ChenPanda/Desktop/pyqt/final/vgg16_model.pth"



    # model1 = torch.load(model1_path, map_location='cpu')
    # root = Node('root')
    # makeTree(model1,root,'')
    # all_brelu_modules = []
    # find_brelu_modules(model1,all_brelu_modules)




    # for name in list1:
    #     entity = get_entity(model1,name)
    #     print(entity)

        # for i in range(len(entity)):
        #     if isinstance(child_module, BRelu):
        #         print(entity[1].boundary)




    num_different_params, different_params = count_different_weights(model1_path, model2_path)
    print(f"Number of different parameters: {num_different_params}")
    print("Different parameters:", different_params)
