# encoding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.ResNet import ResNet50
from . import densenet

class ModelFedCon(nn.Module):
    def __init__(self, out_dim, n_classes):
        super(ModelFedCon, self).__init__()
        model = ResNet50()
        self.features = nn.Sequential(*list(model.children())[:-1])
        num_ftrs = model.fc.in_features

        print("features:",self.features)

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, n_classes)

    def _get_basemodel(self, model_name):
        try:
            model = self.model_dict[model_name]
            #print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet50")

    def forward(self, x):
        h = self.features(x)
        # h = h.squeeze()
        h = h.view(-1,2048)
        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return h, x, y


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self,out_dim, out_size, drop_rate=0):
        super(DenseNet121, self).__init__()
        # assert mode in ('U-Ones', 'U-Zeros', 'U-MultiClass')
        self.densenet121 = densenet.densenet121(pretrained=True, drop_rate=drop_rate)
        num_ftrs = self.densenet121.classifier.in_features
        # if mode in ('U-Ones', 'U-Zeros'):
        #     self.densenet121.classifier = nn.Sequential(
        #         nn.Linear(num_ftrs, out_size),
        #         #                nn.Sigmoid()
        #     )
        # elif mode in ('U-MultiClass',):
        #     self.densenet121.classifier = None
        #     self.densenet121.Linear_0 = nn.Linear(num_ftrs, out_size)
        #     self.densenet121.Linear_1 = nn.Linear(num_ftrs, out_size)
        #     self.densenet121.Linear_u = nn.Linear(num_ftrs, out_size)
        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

        # last layer
        self.l3 = nn.Linear(out_dim, out_size)

        # self.mode = mode

        # Official init from torch repo.
        for m in self.densenet121.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        self.drop_rate = drop_rate
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x):
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        if self.drop_rate > 0:
            out = self.drop_layer(out)
        # self.activations = out
        # if self.mode in ('U-Ones', 'U-Zeros'):
        #     out = self.densenet121.classifier(out)
        # elif self.mode in ('U-MultiClass',):
        #     n_batch = x.size(0)
        #     out_0 = self.densenet121.Linear_0(out).view(n_batch, 1, -1)
        #     out_1 = self.densenet121.Linear_1(out).view(n_batch, 1, -1)
        #     out_u = self.densenet121.Linear_u(out).view(n_batch, 1, -1)
        #     out = torch.cat((out_0, out_1, out_u), dim=1)
        out = out.view(-1, 1024)
        x = self.l1(out)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)
        return out, x, y

# class ModelFedCon_noheader(nn.Module):
#     def __init__(self, n_classes):
#         super(ModelFedCon_noheader, self).__init__()
#         model = ResNet50()
#         self.features = nn.Sequential(*list(model.children())[:-1])
#         num_ftrs = model.fc.in_features
#
#         self.l3 = nn.Linear(num_ftrs, n_classes)
#
#     def _get_basemodel(self, model_name):
#         try:
#             model = self.model_dict[model_name]
#             #print("Feature extractor:", model_name)
#             return model
#         except:
#             raise ("Invalid model name. Check the config file and pass one of: resnet50")
#
#     def foeward(self,x):
#         h = self.features(x)
#         h = h.squeeze()
#
#         y = self.l3(h)
#         return h, h, y