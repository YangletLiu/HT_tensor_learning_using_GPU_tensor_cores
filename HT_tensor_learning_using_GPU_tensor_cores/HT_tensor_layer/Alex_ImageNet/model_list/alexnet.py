import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), 256 * 5 * 5)
        x = self.classifier(x)
        return x

class AlexNet_ht(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_ht, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        r = 8
        n1_1 = 16
        n2_1 = 16
        n3_1 = 25
        m1_1 = 16
        m2_1 = 16
        m3_1 = 16
        #self.fc1 = tnn.Linear(256*6*6, 4096)
        self.u1_1 = nn.Parameter(torch.randn(n1_1,m1_1,r))
        self.b2_1 = nn.Parameter(torch.randn(r,r,r))
        self.u2_1 = nn.Parameter(torch.randn(n2_1,m2_1,r))
        self.b1_1 = nn.Parameter(torch.randn(r,r))
        self.u3_1 = nn.Parameter(torch.randn(n3_1,m3_1,r))
        
        n1_2 = 16
        n2_2 = 16
        n3_2 = 16
        m1_2 = 16
        m2_2 = 16
        m3_2 = 16

        #self.fc2 = nn.Linear(4096, 4096)
        self.u1_2 = nn.Parameter(torch.randn(n1_2,m1_2,r))
        self.b2_2 = nn.Parameter(torch.randn(r,r,r))
        self.u2_2 = nn.Parameter(torch.randn(n2_2,m2_2,r))
        self.b1_2 = nn.Parameter(torch.randn(r,r))
        self.u3_2 = nn.Parameter(torch.randn(n3_2,m3_2,r))

        # Final layer
        self.layer8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)    
#         self.classifier = nn.Sequential(
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )

    def forward(self, x):
        n1_1 = 16
        n2_1 = 16
        n3_1 = 25
        m1_1 = 16
        m2_1 = 16
        m3_1 = 16
        n1_2 = 16
        n2_2 = 16
        n3_2 = 16
        m1_2 = 16
        m2_2 = 16
        m3_2 = 16
        x = self.features(x)
        x = x.view(x.size(0), 256 * 5 * 5)
        
        batch = x.shape[0]
        x = torch.reshape(x,(batch,n1_1,n2_1,n3_1))
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_1))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_1))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_1))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_1))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_1))
        #tnn.BatchNorm1d(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_2))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_2))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_2))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_2))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_2))
        x = F.relu(x)
        x = self.dropout(x)
        
        x = torch.reshape(x,(batch,4096))
        out = self.layer8(x)
        return out


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    # if pretrained:
    #     model_path = 'model_list/alexnet.pth.tar'
    #     pretrained_model = torch.load(model_path)
    #     model.load_state_dict(pretrained_model['state_dict'])
    return model

def ht(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet_ht(**kwargs)
    # if pretrained:
    #     model_path = 'model_list/alexnet.pth.tar'
    #     pretrained_model = torch.load(model_path)
    #     model.load_state_dict(pretrained_model['state_dict'])
    return model
