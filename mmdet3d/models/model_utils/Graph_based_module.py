import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
    '''Position attention mudule'''
    def __init__(self,in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//4, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):#输入Batch X N X Feature 8*128*256
        m_matchsize, C, N = x.size()
        proj_query = self.query_conv(x).permute(0,2,1)#B*N*C
        proj_key = self.key_conv(x)#B*C*N
        energy = torch.bmm(proj_query, proj_key)#B*N*N
        attention = self.softmax(energy)
        proj_value = self.value_conv(x)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = self.gamma*out +x
        return out
class CAM_Module(nn.Module):
    '''Channel attention module'''
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
    def forward(self, x):
        m_batchsize, C, N = x.shape
        # proj_query = self.query_conv(x)
        # proj_key = self.key_conv(x).permute(0,2,1)
        proj_query = x
        proj_key = x.permute(0,2,1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        #proj_value = self.value_conv(x)
        proj_value = x

        out = torch.bmm(attention, proj_value)

        out = self.gamma * out + x
        return out
class DANetHead(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels //4
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)

        self.conva = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 1, bias=False), nn.ReLU())
        self.convc = nn.Sequential(nn.Conv1d(in_channels, inter_channels, 1, bias=False), nn.ReLU())
        self.conva1 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 1, bias=False), nn.ReLU())
        self.convc1 = nn.Sequential(nn.Conv1d(inter_channels, inter_channels, 1, bias=False), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        self.conv2 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(inter_channels, out_channels, 1))
        self.conv3 = nn.Sequential(nn.Dropout(0.1, False), nn.Conv1d(out_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conva(x)
        sa_feat = self.sa(feat1)
        sa_feat = self.conva1(sa_feat)
        sa_output = self.conv1(sa_feat)

        feat2 = self.convc(x)
        sc_feat = self.sc(feat2)
        sc_feat = self.convc1(sc_feat)
        sc_output = self.conv2(sc_feat)
        feat_sum = sa_output + sc_output
        sasc_output = self.conv3(feat_sum)
        return sasc_output

class SENet(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(SENet, self).__init__()
        #SE layers
        self.fc1 = nn.Conv1d(planes, planes//8, kernel_size=1)
        self.fc2 = nn.Conv1d(planes//8, planes, kernel_size=1)       #输入为8*128*256
        if stride !=1 or in_planes!= planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, padding=1, bias=False),
                                          )
    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        #Squeeze
        w = F.avg_pool1d(x, x.size(2))#8*128*256 ->8*128*1
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        x = x*w
        x = x+shortcut
        return x


if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_x = torch.randn(size=(16,128,256)).to(device)
    PAM = PAM_Module(128).to(device)
    CAM = CAM_Module(128).to((device))
    out = PAM(input_x)
    out = CAM(out)
    DANet = DANetHead(128,128).to(device)
    out = DANet(out)
    SENet = SENet(128, 128).to(device)
    out = SENet(out)
