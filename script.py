import cv2
import numpy as np
# a=np.array((1,2,3,4,5,6))
# a=a.reshape(3,2)
# print(a[...,1:2].shape)
'''
import torch
def resize_tensor(x):
    temp = x.squeeze(0).permute(1,2,0).numpy()
    x = np.zeros((224,224,temp.shape[-1]))
    x = cv2.resize(temp, (224,224), interpolation=cv2.INTER_NEAREST)
    x = torch.tensor(x).permute(2,0,1).unsqueeze(0)
resize_tensor(torch.randn(1,20,1000,1000))
ft_size=int(pow(features[i].shape[1],0.5))#zengen
features[i]=torch.reshape(features[i],(features[i].shape[0],ft_size,ft_size,-1)).permute(0,3,1,2)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
class GELU(nn.Module):#zengen
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        #return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# m = GELU()
# IN = torch.tensor((1.0,2.0))
# n = nn.GELU()
# out1 = m(IN)
# out2 = n(IN)
# print(out1,out2)

class Solution:
    def countNegatives(self, grid) -> int:
        ret = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] < 0:
                    ret += len(grid[0]) - j
                    print(j)
                    break
        return ret
m = Solution()
inp = [[4,3,2,-1],[3,2,1,-1],[1,1,-1,-2],[-1,-1,-2,-3]]
ret = m.countNegatives(grid=inp)
print(ret)