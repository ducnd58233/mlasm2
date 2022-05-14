import torch
import torch.nn as nn
from typing import List, Union, Optional, Type, Callable, Any
from torch import Tensor

__all__ = ['resnet10', 'resnet18']

        
        
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        downsample: Optional[nn.Module] = None, 
        stride: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 =  nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 =  nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        
        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out           
        
class ResNet(nn.Module):
    def __init__(
        self, 
        ResBlock: Type[Union[BasicBlock]], 
        layers: List[int], 
        num_classes: int = 4, 
        num_channels: int = 3,
    ):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, channels=64, blocks=layers[0])
        self.layer2 = self._make_layer(ResBlock, channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(ResBlock, channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(ResBlock, channels=512, blocks=layers[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _make_layer(self, 
                    ResBlock: Type[Union[BasicBlock]], 
                    channels: int,
                    blocks: int, 
                    stride: int = 1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != channels * ResBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(channels*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, channels, downsample=downsample, stride=stride))
        self.in_channels = channels * ResBlock.expansion
        
        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, channels))
            
        return nn.Sequential(*layers)
    
def _resnet(
    ResBlock: Type[Union[BasicBlock]],
    layers: List[int],
    num_classes: int = 4, 
    num_channels: int = 3,
    **kwargs: Any,
):
    return ResNet(ResBlock, layers, num_classes, num_channels, **kwargs)

def resnet10(num_classes, num_channels=3):
    return _resnet(BasicBlock, [1, 1, 1, 1], num_classes, num_channels)
    
def resnet18(num_classes, num_channels=3):
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, num_channels)
