import torch.nn as nn
from .utils import IntermediateLayerGetter
from .deeplab import DeepLabHeadV3Plus, DeepLabV3
from .mtss import Mtss, MtssHead, ElevationProcess
from .mtvc import EmbeddingResNet, EmbeddingNet, SiameseNet, TripletNet, ClassificationNet
from .backbone import mobilenetv2
from .backbone import resnet

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    """
    基于resnet的语义分割
    
    Parameters
    ----------
    name: str
        {"deeplabv3plus", "mtss"}optional
        deeplabv3plus：deeplabv3plus
        mtss：多模态地形要素语义分割
    backbone_name: str
        {"resnet50", "resnet101"}optional
    numclass: int
        需要分类的种类数量
    output_stride: int
        输出特征图与输入特征图之间的比值
    pretrained_backbone: bool
        是否使用预训练模型
    """
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    
    if name=='deeplabv3plus':
        # 单遥感图像
        classifier_segm = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        # 修改网络输入为4通道
        backbone.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3(backbone, classifier_segm)
    else:
        # 多模态输入
        elvational_planes = 24
        elevation = ElevationProcess()
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        classifier_segm = MtssHead(inplanes ,low_level_planes, elvational_planes, num_classes, aspp_dilate)
        model = Mtss(backbone, elevation, classifier_segm)
    return model

def _segm_mobilenet(name, backbone, num_classes, output_stride, pretrained_backbone):
    """
    基于mobilenet的语义分割
    
    Parameters
    ----------
    name: str
        {"deeplabv3plus","mtss"}optional
        deeplabv3plus：deeplabv3plus
        mtss：多模态地形要素语义分割
    numclass: int
        需要分类的种类数量
    output_stride: int
        输出特征图与输入特征图之间的比值
    pretrained_backbone: bool
        是否使用预训练模型
    """
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None
    inplanes = 320
    low_level_planes = 24
    return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    if name=='deeplabv3plus':
        # 单遥感图像
        classifier_segm = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        model = DeepLabV3(backbone, classifier_segm)
    else:
        # 多模态输入
        elvational_planes = 24
        elevation = ElevationProcess()
        classifier_segm = MtssHead(inplanes ,low_level_planes, elvational_planes, num_classes, aspp_dilate)
        model = Mtss(backbone, elevation, classifier_segm)
    return model
    
def _load_segmModel(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    """
    地理要素语义分割模型加载
    Parameters
    ----------
    arch_type: str
        {"deeplabv3plus", "mtss"}optional
    backbone: str
        {"resnet50", "resnet101", "mobilenet"}optional
    num_class: int
    output_stride: int
        输入特征图与输出特征图的大小比值
    pretrained_backbone: bool
        是否使用ImageNet预训练模型
    """
    if arch_type == "deeplabv3plus":
        if backbone == 'mobilenetv2':
            model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        elif backbone.startswith('resnet'):
            model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        else:
            raise NotImplementedError
    elif arch_type == "mtss":
        if backbone == "mobilenetv2":
            model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
        elif backbone.startswith('resnet'):
            model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)  
        else:
            raise NotImplementedError
    return model

def _load_ClasModel(arch_type, embedding_name, num_classes):
    """
    村落分类模型加载
    """
    # create embedding_net
    if embedding_name == 'embeddingNet':
        embedding_net = EmbeddingNet()
    elif embedding_name == 'embeddingResNet':
        embedding_net = EmbeddingResNet()
    else:
        raise ValueError("embedding_name error!Please check and try again!")
    # create model
    if arch_type == 'classificationNet':
        model = ClassificationNet(embedding_net, num_classes)
    elif arch_type == 'siameseNetwork':
        model = SiameseNet(embedding_net)
    elif arch_type == 'tripletNetwork':
        model = TripletNet(embedding_net)
    elif arch_type == 'onlinePairSelection':
        model = embedding_net
    elif arch_type == 'onlineTripletSelection':
        model = embedding_net
    else:
        raise ValueError("arch_type error!Please check and try again!")
    return model

# Segmentation model
# Mtss Baseline: Deeplab v3+
def deeplabv3plus_resnet50(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# mtss
def mtss_resnet50(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a Mtss model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('mtss', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def mtss_resnet101(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a Mtss model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('mtss', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def mtss_mobilenet(num_classes=7, output_stride=16, pretrained_backbone=True):
    """Constructs a Mtss model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_segmModel('mtss', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

# Classification model
# Mtvc Baseline: classification with softmax
def classificationNet(embedding_name, num_classes=6):
    """Constructs a Mtvc classification with softmax"""
    return _load_ClasModel('classificationNet', embedding_name, num_classes=num_classes)

# siamese
def siameseNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('siameseNetwork', embedding_name, num_classes=num_classes)

# triplet
def tripletNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('tripletNetwork', embedding_name, num_classes=num_classes)

# onlinePairSelection
def onlinePairSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlinePairSelection', embedding_name, num_classes=num_classes)

# onlineTripletSelection
def onlineTripletSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlineTripletSelection', embedding_name, num_classes=num_classes)

if __name__ == "__main__":
    import torch
    a = torch.rand(1, 3, 224, 224)
    net = deeplabv3plus_mobilenet(7, 16, False)
    print(net)