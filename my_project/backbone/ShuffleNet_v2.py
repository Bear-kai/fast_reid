import torch
import torch.nn as nn
from collections import OrderedDict

class ShuffleV2Block(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2        # 保证stride=1时特征图的size不变
        self.pad = pad
        self.inp = inp

        outputs = oup - inp     # branch_main输出outputs个通道 + branch_proj输出inp个通道，cat后就是输出oup个通道

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert (num_channels % 4 == 0)
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)     # 这种通过reshape+permute再切片的方式，应该比直接切片更高效
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    """
    Reference:
        https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV2
    """

    def __init__(self, input_size=224, n_class=1000, model_size='0.5x', feat_dim=1024):
        super(ShuffleNetV2, self).__init__()
        print('model size is ', model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, feat_dim]   # 将1024替换为参数feat_dim
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, feat_dim]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, feat_dim]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, feat_dim*2]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    self.features.append(ShuffleV2Block(input_channel, output_channel,          # stride=2时输入通道数不变(为上一级输出通道数)
                                                        mid_channels=output_channel // 2, ksize=3, stride=2))
                else:
                    self.features.append(ShuffleV2Block(input_channel // 2, output_channel,     # stride=1时输入通道被shuffle减半
                                                        mid_channels=output_channel // 2, ksize=3, stride=1))

                input_channel = output_channel

        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True)
        )
        self.globalpool = nn.AvgPool2d(7)
        if self.model_size == '2.0x':
            self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)
        if self.model_size == '2.0x':
            x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])   # 对应何种初始化方式? 待续...
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ShuffleNetV2Backbone(nn.Module):
    def __init__(self, model_size, pretrained=False, pretrain_path='', **args):
        super(ShuffleNetV2Backbone, self).__init__()

        model = ShuffleNetV2(model_size=model_size, feat_dim=args['feat_dim'])
        if pretrained:
            new_state_dict = OrderedDict()
            state_dict = torch.load(pretrain_path)['state_dict']
            for k, v in state_dict.items():
                if k[:7] == 'module.':
                    k = k[7:]
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=True)

        if args['pool_layer'] == 'pool_s2':
            self.backbone = nn.Sequential(
                model.first_conv, model.maxpool, model.features, model.conv_last)
        elif args['pool_layer'] == 'pool_s1':
            maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
            self.backbone = nn.Sequential(
                model.first_conv, maxpool, model.features, model.conv_last)
        elif args['pool_layer'] == 'no_pool':
            self.backbone = nn.Sequential(
                model.first_conv, model.features, model.conv_last)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ShuffleNetV2()
    # print(model)

    # test_data = torch.rand(5, 3, 224, 224)
    test_data = torch.rand(5, 3, 256, 128)
    test_outputs = model(test_data)
    print(test_outputs.size())