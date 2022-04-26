import torch.nn as nn
import torchvision
import torch.nn.functional as F
from .models import register


class Block(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=2):
        padding = (kernel_size - 1) // 2
        super(Block, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_planes),
            nn.MaxPool2d((stride, stride))
        )


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        net_para = [
            # i, o, k, s
            [1, 16, 3, 2],
            [16, 32, 3, 2],
            [32, 64, 3, 2],
            [64, 32, 3, 2],
        ]
        features = []
        for i, o, k, s in net_para:
            features.append(Block(i, o, k, s))

        # self.up = nn.Upsample((100,100))
        self.features = nn.Sequential(*features)
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 20)
        )

    def forward(self, x):
        # x = self.up(x)
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EmbeddingResNet, self).__init__()
        model_resnet18 = torchvision.models.resnet18(pretrained=True)
        # 更改resnet18第一层的输入通道数
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features
        self.fc = nn.Linear(self.__in_features, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


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


# Classification model
# Mtvc Baseline: classification with softmax
@register('classificationNet')
def classificationNet(embedding_name, num_classes=6):
    """Constructs a Mtvc classification with softmax"""
    return _load_ClasModel('classificationNet', embedding_name, num_classes=num_classes)


# siamese
@register('siameseNetwork')
def siameseNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('siameseNetwork', embedding_name, num_classes=num_classes)


# triplet
@register('tripletNetwork')
def tripletNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('tripletNetwork', embedding_name, num_classes=num_classes)


# onlinePairSelection
@register('onlinePairSelection')
def onlinePairSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlinePairSelection', embedding_name, num_classes=num_classes)


# onlineTripletSelection
@register('onlineTripletSelection')
def onlineTripletSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlineTripletSelection', embedding_name, num_classes=num_classes)

