
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.SyncBatchNorm(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class ConvolutionalAttention(nn.Module):
    """
    The ConvolutionalAttention implementation
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.SyncBatchNorm(in_channels)

        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 3, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 3))



    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)
        x = x.reshape([x_shape[0], self.inter_channels, h, w])
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): The input tensor. (n,c,h,w)
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        
        # **保证 H 和 W 为偶数，以避免 Conv2d 计算时尺寸错位**
        if x.shape[2] % 2 != 0:  # 高度 H
            x = x[:, :, :-1, :]
        if x.shape[3] % 2 != 0:  # 宽度 W
            x = x[:, :, :, :-1]

        x1 = F.conv2d(
                x,
                self.kv,
                bias=None,
                stride=1,
                padding=(1,0))
        x1 = self._act_dn(x1)
        x1 = F.conv2d(
                x1, self.kv.transpose(1, 0), bias=None, stride=1,
                padding=(1,0))
        x3 = F.conv2d(
                x,
                self.kv3,
                bias=None,
                stride=1,
                padding=(0,1))
        x3 = self._act_dn(x3)
        x3 = F.conv2d(
                x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,1))
        # 确保 x1 和 x3 形状匹配
        min_h = min(x1.shape[2], x3.shape[2])  # 找到最小的高度
        min_w = min(x1.shape[3], x3.shape[3])  # 找到最小的宽度
        x1 = x1[:, :, :, :min_w]  # 截取
        x3 = x3[:, :, :, :min_w]
        
        x=x1+x3
        return x

class CMBlock(nn.Module):
    """
    The CMBlock implementation based on PaddlePaddle.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in CMBlock. Default: 0.2
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.1):
        super(CMBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=64,
            num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.attention_weights = None  # 存储原始注意力权重
        self.feature_maps = []        # 存储特征图
 
    def forward(self, x):
        x_res = x
        attn_out = self.attn_l(x)           # 原先直接 x_res + self.drop_path(self.attn_l(x))
        # 保存可视化数据
        self.attention_weights = self.attn_l.attn_map  # 从ConvolutionalAttention获取
        self.feature_maps = {
            'input': x_res.detach().cpu(),
            'output': attn_out.detach().cpu()
        }
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x))
        return x

class CM_UNet(nn.Module):
    def __init__(self, in_nc=3, config=[2,2,2,2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=256):
        super(CM_UNet, self).__init__()
        self.config = config
        self.dim = dim

        # Drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        self.m_head = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]

        begin = 0
        self.m_down1 = [CMBlock(dim, dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [CMBlock(2*dim, 2*dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [CMBlock(4*dim, 4*dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [CMBlock(8*dim, 8*dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[3])]

        begin += config[3]
        self.m_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                     [CMBlock(4*dim, 4*dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[4])]

        begin += config[4]
        self.m_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                     [CMBlock(2*dim, 2*dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[5])]

        begin += config[5]
        self.m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                     [CMBlock(dim, dim, num_heads=8, drop_rate=0., drop_path_rate=dpr[i+begin])
                      for i in range(config[6])]

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):
        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]
        return x

    


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



if __name__ == '__main__':
    # 实例化 CM-UNet 网络
    net = CM_UNet(in_nc=3, input_resolution=128)

    # 创建一个 2x3x128x128 的假输入（batch size 2, 通道数 3, 图像尺寸 128x128）
    x = torch.randn((2, 3, 128, 128))

    # 前向传播
    output = net(x)

    # 打印输出形状
    print("Output shape:", output.shape)
