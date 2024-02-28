import torch
from torch import tensor
item = tensor([[[[3, 12, -4, 8],
            [15, 7, 2, -6],
            [10, -2, 9, 11]],

           [[-1, 4, 13, 6],
            [5, 8, 1, -3],
            [7, -5, 14, 10]]],

          [[[11, 2, 8, -9],
            [4, -1, 6, 15],
            [0, 9, 12, 3]],

           [[6, 3, -7, 5],
            [2, 10, -4, 8],
            [13, 7, 1, -2]]]])



max_list = [-10000]

m = 0
n = 0

print(item[m,n,:,:].flatten())
out_flat=torch.topk(item[m,n,:,:].flatten(),int(0.8*len(item[m,n,:,:].flatten())),largest=False)
print(out_flat)
max_num=out_flat.values.max()
max_list = max(max_num.item(),max_list[0])

print(max_list)