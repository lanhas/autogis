import torch.nn as nn
import torchvision
import torch.nn.functional as F


class Mtvc():
    """
    Multimodal traditional village classification(Mtvc) is a network to classify traditional village
    which has specific landscape relationship in China.It carries out classify by sending remote sensing
    image and DEM data into a sementic segmentation network named Mtsn to get the feature of landscape
    elements.Then, we classify the feature through the classification network to get the village type.

    多模态传统村落分类网络是一个根据中国传统村落特殊的山水关系对村落类型进行分类的网络，它通过将村落的遥感影像和
    DEM数据送入一个Mtts语义分割网络得到山水要素的特征图，随后将其送入分类网络得到村落类型

    Arguments:
        embedding(nn.Module):the network used to extract the represtation from feature map
        n_classed: the number of classes
    """
    pass


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


class _block(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=2):
        padding = (kernel_size - 1) // 2
        super(_block, self).__init__(
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
            features.append(_block(i, o, k, s))

        # self.up = nn.Upsample((100,100))
        self.features = nn.Sequential(*features)
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
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
