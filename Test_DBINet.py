import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.DBINet import DBINet


if __name__ == "__main__":
    print('Test DBINet !')

    input=torch.rand(2,3,512,512).cuda()
    print('input.shape:',input.shape)


    model=DBINet().cuda()
    output=model(input)

    for out in output:
        print('out.shape:',out.shape)




