import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# 传统语义分割网络入口
class _SimpleSegmentationModel(nn.Module):
    """
    将语义分割网络变成backbone和classifier两部分，也是model的入口部分
    """
    def __init__(self, backbone, segmention):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.segmention = segmention
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.segmention(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


# 地理信息要素的多模态语义分割网络入口
class _MultimodelSegmentationModel(nn.Module):
    """
    多模态语义分割网络，用于地理信息要素的语义分割，输入为遥感图像和高程数据，
    """
    def __init__(self, backbone, elevation, segmention):
        super(_MultimodelSegmentationModel, self).__init__()
        self.backbone = backbone
        self.elevation  = elevation
        self.segmention = segmention
    def forward(self, x, y):
        # x:遥感数据 y:高程数据
        input_shape = x.shape[-2:]
        features_rs = self.backbone(x)
        features_el = self.elevation(y)
        x = self.segmention(features_rs, features_el)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x


# 主干特征提取网络入口
class IntermediateLayerGetter(nn.ModuleDict):
    """
    得到主干网络
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        # 将return_layers保存成字典
        return_layers = {k: v for k, v in return_layers.items()}
        # 构造一个有序字典，保存非return_layer的部分
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
    
    def forward(self, x):
        out = OrderedDict()
        # 
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
