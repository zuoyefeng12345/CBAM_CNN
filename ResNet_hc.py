import torch
import torch.nn as nn
import math
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3卷积 with padding"""
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=dilation,
                     dilation=dilation, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

#  空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 普通Block模块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64!")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock!")
        # When the stride step is not 0, self.conv1 and self.downsample the input x
        self.conv1 = conv3x3(in_planes, planes, stride)  # When stride!=0, there will be a change to the
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride  #

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# Bottleneck模块
class Bottleneck(nn.Module):
    expansion = 4  #

    # The bottleneck here uses conv2 for downsampling, whereas conv1 is used for downsampling in native ResNet, which can further improve accuracy
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:  # If no BatchNorm layer is passed in, directly set the BatchNorm
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 用于 ResNext 或 Wide-ResNetFor ResNext or Wide-ResNet
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=1000, zero_init_residual=False, init_weight=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # Each element in the tuple represents whether to replace the corresponding 2x2 stride with a null convolution
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True);

        # Attention mechanism added to the first layer of the network
        # self.ca = ChannelAttention(self.inplanes)
        # self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # The last layer of the convolutional layer of the network incorporates an attention mechanism
        # self.ca1 = ChannelAttention(self.inplanes)
        # self.sa1 = SpatialAttention()


        # The last layer is the fully connected layer, in order to migrate the learning to be able to use the official pre-training mods provided
        self.fc = nn.Linear(512 * block.expansion, num_class)

        if init_weight:
            for m in self.modules():  # initialization
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # Initialize 0 for the last bn layer of each residual module
        # This has the advantage of making each residual module similar to a constant connection at the very beginning, thus further improving performance
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, num_classes, **kwargs):  # The official code here calls pretrained model, we call outside the
    model = ResNet(block, layers, num_class=num_classes, **kwargs)
    return model


 #ResNet-18
def resnet18(num_classes=1000, **kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


# ResNet-34
def resnet34(num_classes=1000, **kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes, **kwargs)


#ResNet-50
def resnet50(num_classes=1000, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


# ResNet-101
def resnet101(num_classes=1000, **kwargs):
   return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


# ResNet-152
def resnet152(num_classes=1000, **kwargs):
   return _resnet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs)


if __name__ == '__main__':
   torch.manual_seed(1)
   Model = resnet101(2)
   print(Model)