import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class GlobalResidualNet(nn.Module):
    def __init__(self,global_in_channel=512,
                 global_out_channel=128):
        super(GlobalResidualNet, self).__init__()
        self.gs_conv1 = torch.nn.Conv1d(global_in_channel, global_out_channel, 1)
    def forward(self, features, seed_features, net):#features(B,128,256)  seed_features(B*128*K)
        global_features_1 = F.max_pool1d(features, kernel_size=features.size(2))#(B, 128, 1)
        global_features_2 = F.max_pool1d(seed_features, kernel_size=seed_features.size(2))#(B*256*1)
        global_features = torch.cat((global_features_2,global_features_1),1)#(B,256+128,256)
        global_features = torch.cat((global_features.expand(features.shape[0],256+128,256),net),1)
        global_features = self.gs_conv1(global_features)
        global_features = torch.sigmoid(torch.log(torch.abs(global_features)))
        net = net * global_features
        return net