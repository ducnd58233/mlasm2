import torch
import torch.nn as nn
from typing import Dict, Union, List, Any, cast

__all__ = ['vgg16', 'vgg16_bn']

class VGG(nn.Module):
    r"""
    Desc:
        Customize for Machine Learning Asm2
    
    Args:
        features: Using make_layers function to config blocks
        num_classes (int): The number of classes (output)
        init_weights (bool): If true, create weights
    """
    
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 2,
        init_weights: bool = True,
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.max_pool = nn.AdaptiveMaxPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)           
        )
        if init_weights:
            self._init_weights()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    r"""
    Desc:
        Make layers for VGG
        
    Args:
        cfg (List): config types define in cfgs
        batch_norm (bool): If true, using batch normalization. Default: False
        in_channels (int): Number of channels of the input. Default: 3 
    """
    
    layers: List[nn.Module] = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:    
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[int, List[Union[str, int]]] = {
    'A': [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'B': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
}

def _vgg(cfg: str, batch_norm: bool, num_classes: int = 2, in_channels: int = 3, init_weights: bool = True, **kwargs: Any) -> VGG:
    r"""
    Args:
        cfg (str): config types defines in cfgs
        batch_norm (bool): If true, model will use batch normalization
        num_classes (int): Number of classes (output). Default: 2
        in_channels (int): Number of channels of the input. Default: 3 
    """
    
    return VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels), num_classes=num_classes,  **kwargs)

def vgg8(num_classes: int = 2, in_channels: int = 3, init_weights: bool = True, **kwargs):
    r"""
    Desc: 
        VGG8 without batch normalization.
    
    Args:
        num_classes (int): Number of classes (output). Default: 2
        in_channels (int): Number of channels of the input. Default: 3 
        init_weights (bool): If true, create weights. Default: True
    """
    
    return _vgg('B', False, num_classes, in_channels, init_weights, **kwargs)

def vgg8_bn(num_classes: int = 2, in_channels: int = 3, init_weights: bool = True, **kwargs):
    r"""
    Desc: 
        VGG8 with batch normalization.
    
    Args:
        num_classes (int): Number of classes (output). Default: 2
        in_channels (int): Number of channels of the input. Default: 3 
        init_weights (bool): If true, create weights. Default: True
    """
    
    return _vgg('B', True, num_classes, in_channels, init_weights, **kwargs)
    
def vgg16(num_classes: int = 2, in_channels: int = 3, init_weights: bool = True, **kwargs):
    r"""
    Desc: 
        VGG16 without batch normalization.
    
    Args:
        num_classes (int): Number of classes (output). Default: 2
        in_channels (int): Number of channels of the input. Default: 3 
        init_weights (bool): If true, create weights. Default: True
    """
    
    return _vgg('A', False, num_classes, in_channels, init_weights, **kwargs)

def vgg16_bn(num_classes: int = 2, in_channels: int = 3, init_weights: bool = True, **kwargs):
    r"""
    Desc: 
        VGG16 with batch normalization.
    
    Args:
        num_classes (int): Number of classes (output). Default: 2
        in_channels (int): Number of channels of the input. Default: 3 
        init_weights (bool): If true, create weights. Default: True
    """
    
    return _vgg('A', True, num_classes, in_channels, init_weights, **kwargs)