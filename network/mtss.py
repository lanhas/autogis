import torch
from torch import nn
from torch.nn import functional as F
from network.utils import _MultimodelSegmentationModel

__all__ = ["Mtss", "MtssHead", "ElevationProcess", "convert_to_separable_conv"]


class Mtss(_MultimodelSegmentationModel):
    """
    Multimodel terrain Semantic segmentation(Mtss) is a network to Semantic 
    segmentation of terrain elements.It carries out semantic segmentation by sending 
    remote sensing image and DEM data into network together.

    多模态双阶段地形要素语义分割网络是一个针对地理信息要素设计的语义分割网络，它利用遥感影像
    和DEM数据共同构成特征进行地形要素识别

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        elevation (nn.Module): module that take the "out" element return from backbone
            and elevation return a sense prediction.
        segmentation (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.

        backbone (nn.Module): 主干网络用于从遥感影像计算特征，它返回一个有序字典OrderedDict[Tensor]，
            键值对为out表示最后使用的特征映射，aux为辅助分类器
        elevation (nn.Module): 高程网络用于从DEM中计算特征，辅助主干网络进行要素划分
        segmentation (nn.Module): 将主干网络和高程网络进行特征融合从而进行分类
    """
    pass


class MtssHead(nn.Module):
    def __init__(self, in_channels, low_level_channels, elvation_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(MtssHead, self).__init__()
        # 对提取的低层特征进行卷积
        self.project_rs = nn.Sequential( 
            nn.Conv2d(low_level_channels, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        # 对提取的高程特征进行卷积
        self.project_el = nn.Sequential(
            nn.Conv2d(elvation_channels, 24, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        # 空洞空间卷积池化金字塔(atrous spatial pyramid pooling (ASPP))
        self.aspp = ASPP(in_channels, aspp_dilate)
        # 分类器
        self.classifier_segm = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature_rs, feature_el):
        low_level_feature = self.project_rs(feature_rs['low_level'])
        elvation_feature = self.project_el(feature_el)
        output_feature = self.aspp(feature_rs['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier_segm(torch.cat([ low_level_feature, elvation_feature, output_feature ], dim=1 ) )
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ElevationProcess(nn.Module):
    """
    高程特征提取网络
    """
    def __init__(self):
        super(ElevationProcess,self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(1, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
