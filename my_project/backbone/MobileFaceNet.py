#coding=utf-8
from torch import nn
import torch
from torch.autograd import Variable
import math

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from fastreid.layers import AdaptiveAvgMaxPool2d, AdaptiveAvgMaxPool2d_cat

__all__ = ['MobileFaceNet', 'MobileFaceNet_air']

# input_size = 112x112
MobileFaceNet_Setting = {
                        'MobileFaceNet':[
                            # t, c , n ,s
                            [2, 64, 5, 2],                            # 5=1+4, 后面4个块输入输出都是28x28
                            [4, 128, 1, 2],
                            [2, 128, 6, 1],
                            [4, 128, 1, 2],
                            [2, 128, 2, 1]
                        ] ,                                            # MobileFaceNet算上最后两个linear conv，共有50层
                        'MobileFaceNet2x_1':[
                            # t, c , n ,s
                            [2, 64, 9, 2],       # in 56 out 28       # 9=1+8
                            [4, 128, 1, 2],      # in 28 out 14
                            [2, 128, 16, 1],     # in 14 out 14
                            [4, 128, 1, 2],      # in 14 out 7
                            [2, 128, 4, 1]       # in 7 out 7
                        ],                                             # 如此修改，MobileFaceNet共有50+(4+10+2)*3=98层 --> 同insightface中的mobilefacenet-y2参数设置
                        'MobileFaceNet2x_2':[
                            # t, c , n ,s
                            [2, 64, 5, 2],       # in 56 out 28
                            [4, 128, 1, 2],      # in 28 out 14
                            [2, 128, 23, 1],     # in 14 out 14      # 仿照ResNet50 --> ResNet101，仅对14x14的block加倍
                            [4, 128, 1, 2],      # in 14 out 7
                            [2, 128, 2, 1]       # in 7 out 7
                        ]                                            # 如此修改，MobileFaceNet共有50+(23-6)*3=101层
}

# original, input_size=112x112
# Mobilefacenet_setting_air = [
#     # t, c , n ,s
#     [2, 64, 10, 2],                          # 10=1+9, 后面9个块输入输出都是28x28
#     [4, 128, 1, 2],
#     [2, 128, 16, 1],
#     [8, 256, 1, 2],
#     [2, 256, 6, 1]
# ]

# modified, for person reid, input_size=256x128
Mobilefacenet_setting_air = [
    # t, c , n ,s
    [2, 64, 10, 2],                          # 10=1+9, 后面9个块输入输出都是28x28
    [4, 128, 1, 2],
    [2, 128, 12, 1],
    [4, 256, 1, 2],
    [2, 256, 4, 1]
]

# # input_size = 224x224
Mobilenetv2_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class CBAM_module_channel(nn.Module):
    def __init__(self, channels, reduction, tanh_act):
        super(CBAM_module_channel, self).__init__()
        self.tanh_act = tanh_act
        self.avg_pool = nn.AdaptiveAvgPool2d(1)     # https://blog.csdn.net/u013382233/article/details/85948695
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.fc1.weight.data)
        nn.init.xavier_uniform_(self.fc2.weight.data)

    def forward_MLP(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        module_input = x
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x_avg = self.forward_MLP(x_avg)
        x_max = self.forward_MLP(x_max)
        x_merge = x_avg + x_max
        if self.tanh_act:
            Mc = self.tanh(x_merge) + 1    # 参airFace中的设置，2019.8.19 add
        else:
            Mc = self.sigmoid(x_merge)
        return module_input * Mc


class CBAM_module_spatial(nn.Module):
    def __init__(self, tanh_act):
        super(CBAM_module_spatial, self).__init__()
        self.tanh_act = tanh_act
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        nn.init.xavier_uniform_(self.conv.weight.data)

    def forward(self, x):
        module_input = x                             # (N,C,H,W)
        x_avg = torch.mean(x,1,keepdim=True)        # (N,1,H,W)
        x_max = torch.max(x,1, keepdim=True)[0]     # Returns a namedtuple (values, indices)
        x_merge = torch.cat((x_avg, x_max),dim=1)   # (N,2,H,W)
        x_merge = self.conv(x_merge)                # (N,1,H,W)
        if self.tanh_act:
            Ms = self.tanh(x_merge) + 1             # 参airFace中的设置，2019.8.19 add
        else:                                      # tanh_act貌似趋向差异更大(接近0的)，初始运行结果摘录Ms[0,0,0,-4:]：
            Ms = self.sigmoid(x_merge)              # tanh_act:  5.3644e-07, 6.4373e-06, 4.2450e-04, 1.5980e-04 ...
        return module_input * Ms                   # sigmoid: 0.0005, 0.0018, 0.0144, 0.0089 ...


class CBAM_module_combine(nn.Module):
    def __init__(self,channels, reduction, tanh_act):
        super(CBAM_module_combine,self).__init__()
        self.channel_att = CBAM_module_channel(channels, reduction, tanh_act)
        self.spatial_att = CBAM_module_spatial(tanh_act)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion, CBAM=False, tanh_act=False):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup       # 须输入输出通道数一致，且步长为1(表示feature_size一致)时，启动跳跃连接
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        if CBAM:
            self.conv.add_module(str(len(self.conv)), CBAM_module_combine(oup, 16, tanh_act))  # 参Sequential定义编写

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class MobileFaceNet(nn.Module):
    def __init__(self, setting_str='MobileFaceNet', CBAM=False, 
                 tanh_act=False, allow_dim_2x=True, pooltype='GDConv', l7scale=1):
        super(MobileFaceNet, self).__init__()
        self.cbam = CBAM
        self.tanh_act = tanh_act
        bottleneck_setting = MobileFaceNet_Setting[setting_str]    # 2019.10.9 add

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)                     # conv3x3

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)       # depthwise conv3x3

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)  # 5 bottleneck

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)                  # conv1x1

        # self.linear7 = ConvBlock(512, 512, (7, 7), 1, 0, dw=True, linear=True)  # Global-Depth wise Conv
        l7_scale1 = 1
        l7_scale2 = 1
        self.pooltype = pooltype
        if pooltype == 'GDConv':
            self.linear7 = ConvBlock(512, 512, (16, 8), 1, 0, dw=True, linear=True) 
        elif pooltype == 'GAP':
            self.linear7 = nn.AdaptiveAvgPool2d(1) 
        elif pooltype == 'GMP':
            self.linear7 = nn.AdaptiveMaxPool2d(1)
        elif pooltype == 'GAMP_add':
            self.linear7 = AdaptiveAvgMaxPool2d()
        elif pooltype == 'GAMP_cat':
            self.linear7 = AdaptiveAvgMaxPool2d_cat()
            l7_scale1 = 2           # 2020.12.8 add l7scale
            l7_scale2 = l7scale

        self.linear1 = ConvBlock(512*l7_scale1, 128*l7_scale2, 1, 1, 0, linear=True)       # linear conv1x1
        if bottleneck_setting[2][2] != 6 and allow_dim_2x:
            self.linear1 = ConvBlock(512*l7_scale1, 256*l7_scale2, 1, 1, 0, linear=True)   # 对应MobileFaceNet_2x, 输出特征维度加倍

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):              # 2019.10.15 add fc_init,不过MobileFaceNet中的确没有nn.Linear层
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t, self.cbam, self.tanh_act))
                else:
                    layers.append(block(self.inplanes, c, 1, t, self.cbam, self.tanh_act))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):           # 3x112x112  -->   256x128
        x = self.conv1(x)           # 64x56x56   -->   128x64
        x = self.dw_conv1(x)        # 64x56x56   -->   128x64
        x = self.blocks(x)          # 128x7x7    -->   16x8
        x = self.conv2(x)           # 512x7x7    -->   16x8
        x = self.linear7(x)         # 512x1x1    -->   1x1
        x = self.linear1(x)         # 128x1x1    -->   1x1
        # x = x.view(x.size(0), -1)   

        return x


class MobileFaceNet_air(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_setting_air, CBAM=False, 
                 tanh_act=False, pooltype='GDConv', l7scale=1):
        super(MobileFaceNet_air, self).__init__()
        self.cbam = CBAM
        self.tanh_act = tanh_act

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)                     # conv3x3

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)        # depthwise conv3x3   注意dw_conv时，输入输出通道一般相等，后接pw_conv改变输出通道，
                                                                   # dw_conv+pw_conv称为深度可分离卷积
        self.inplanes = 64                                         # nn.Conv2d()的分组参数必须能被输入输出通道数整除，比如in=64,out=128，
        block = Bottleneck                                         # 则1个group对应含1个输入通道，对应2个输出通道
        self.blocks = self._make_layer(block, bottleneck_setting)  # 5 bottleneck

        # self.conv2 = ConvBlock(256, 1024, 1, 1, 0)                  # conv1x1
        self.conv2 = ConvBlock(256, 512, 1, 1, 0)                  # conv1x1

        # self.linear7 = ConvBlock(1024, 1024, (7, 7), 1, 0, dw=True, linear=True)  # linear GDConv7x7; 若输入img_size是112*96，则此处kernel_size可改为7*6
        l7_scale1 = 1
        l7_scale2 = 1
        self.pooltype = pooltype
        if pooltype == 'GDConv':
            # self.linear7 = ConvBlock(1024, 1024, (16, 8), 1, 0, dw=True, linear=True)
            self.linear7 = ConvBlock(512, 512, (16, 8), 1, 0, dw=True, linear=True)
        elif pooltype == 'GAP':
            self.linear7 = nn.AdaptiveAvgPool2d(1) 
        elif pooltype == 'GMP':
            self.linear7 = nn.AdaptiveMaxPool2d(1)
        elif pooltype == 'GAMP_add':
            self.linear7 = AdaptiveAvgMaxPool2d()
        elif pooltype == 'GAMP_cat':
            self.linear7 = AdaptiveAvgMaxPool2d_cat()
            l7_scale1 = 2           # 2020.12.8 add l7scale
            l7_scale2 = l7scale

        # self.linear1 = ConvBlock(1024*l7_scale1, 128*l7_scale2, 1, 1, 0, linear=True)  # linear conv1x1 论文的Table 1中是输出512dim，实验中描述是128dim
        self.linear1 = ConvBlock(512*l7_scale1, 128*l7_scale2, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t, self.cbam, self.tanh_act))
                else:
                    layers.append(block(self.inplanes, c, 1, t, self.cbam, self.tanh_act))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):               # 3x256x128
        x = self.conv1(x)               # 64x128x64
        x = self.dw_conv1(x)            # 64x128x64
        x = self.blocks(x)              # 256x16x8
        x = self.conv2(x)               # 1024x16x8
        x = self.linear7(x)             # 1024x1x1
        x = self.linear1(x)             # 128x1x1
        # x = x.view(x.size(0), -1)

        return x


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    input = Variable(torch.FloatTensor(2, 3, 256, 128))
    # net = MobileFaceNet()
    net = MobileFaceNet_air()
    print(net)
    x = net(input)
    print(x.shape)
