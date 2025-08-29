
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeature(nn.Module):
    def __init__(self, num_classes=100, init_weights=False, dropout=0.5):
        super(VGGFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=100, init_weights=False, dropout=0.5):
        super(VGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CompVGGFeature(nn.Module):
    def __init__(self):
        super(CompVGGFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        x = self.features(x)
        return x

class CompVGGClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(CompVGGClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes=100, dropout=0.5):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x_feat = self.avgpool(x)
        x = torch.flatten(x_feat, 1)
        x = self.classifier(x)
        return x_feat, x

class CompVGG(nn.Module):
    def __init__(self, num_classes=100):
        super(CompVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x_feat = self.avgpool(x)
        x = torch.flatten(x_feat, 1)
        x = self.classifier(x)
        return x_feat, x

class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x_feat = self.features(x)
        x = self.avgpool(x_feat)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x_feat, x




# import torch
# import torch.nn as nn
# import torch.nn.functional as F



# class VGGFeature(nn.Module):
#     def __init__(self, num_classes=200, init_weights=False, dropout=0.5):
#         super(VGGFeature, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         return x
    
# class VGGClassifier(nn.Module):
#     def __init__(self, num_classes=200, init_weights=False, dropout=0.5):
#         super(VGGClassifier, self).__init__()

#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
    
# class CompVGGFeature(nn.Module):
#     def __init__(self):
#         super(CompVGGFeature, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=4, stride=4),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#     def forward(self, x):
#         x = self.features(x)
#         return x
    
# class CompVGGClassifier(nn.Module):
#     def __init__(self, num_classes=200):
#         super(CompVGGClassifier, self).__init__()

#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
#     def forward(self, x):
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x




# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.avgpool(x)
#         return x
    

# # class Discriminator(nn.Module):
# #     def __init__(self, in_channels=512):
# #         super(Discriminator, self).__init__()
# #         self.conv = nn.Sequential(
# #             nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
# #             nn.ReLU(True),
# #             nn.Conv2d(256, 128, kernel_size=3, padding=1),
# #             nn.ReLU(True),
# #             nn.AdaptiveAvgPool2d((1, 1)),
# #         )
# #         self.fc = nn.Linear(128, 1)

# #     def forward(self, x):
# #         x = self.conv(x)
# #         x = torch.flatten(x, 1)
# #         x = self.fc(x)
# #         return x

# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()

# #         self.linear = nn.Sequential(
# #             nn.Linear(64 * 2 * 2, 64),
# #             nn.ReLU(True),
# #             nn.Linear(64, 1),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, x):

# #         x = torch.flatten(x, 1)
# #         x = self.linear(x)
# #         return x
# class Discriminator(nn.Module):
#     def __init__(self, input_size=512*7*7):  # 512*7*7=25088
#         super(Discriminator, self).__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.ReLU(True),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = self.linear(x)
#         return x
# class VGG(nn.Module):
#     def __init__(self, num_classes=200, dropout=0.5):
#         super(VGG, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
#             # Remove last maxpool to keep 4x4 resolution
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
        
#         # Keep 4x4 resolution (512*4*4 = 8192)
#         self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 4 * 4, 2048),  # Matches checkpoint [2048, 8192]
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(2048, 1024),          # Matches checkpoint [1024, 2048]
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(1024, num_classes),    # Matches checkpoint [200, 1024]
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x_feat = x
#         x = torch.flatten(x_feat, 1)
#         x = self.classifier(x)
#         return x_feat, x

# # class VGG(nn.Module):
# #     def __init__(self, num_classes=200, dropout=0.5):
# #         super(VGG, self).__init__()
# #         self.features = nn.Sequential(
# #             nn.Conv2d(3, 64, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Conv2d(64, 128, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Conv2d(128, 256, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(256, 256, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Conv2d(256, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.Conv2d(512, 512, kernel_size=3, padding=1),
# #             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
# #         )

# #         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
# #         self.classifier = nn.Sequential(
# #             nn.Linear(512 * 7 * 7, 4096),
# #             nn.ReLU(True),
# #             nn.Dropout(p=dropout),
# #             nn.Linear(4096, 4096),
# #             nn.ReLU(True),
# #             nn.Dropout(p=dropout),
# #             nn.Linear(4096, num_classes),
# #         )

# #     def forward(self, x):
# #         x = self.features(x)
# #         x_feat = self.avgpool(x)
# #         x = torch.flatten(x_feat, 1)
# #         x = self.classifier(x)
# #         return x_feat, x
    

# class CompVGG(nn.Module):
#     def __init__(self, num_classes=200):
#         super(CompVGG, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=4, stride=4),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 7 * 7, 512),
#             nn.ReLU(True),
#             nn.Linear(512, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x_feat = self.avgpool(x)
#         x = torch.flatten(x_feat, 1)
#         x = self.classifier(x)
#         return x_feat, x
    

# class AlexNet(nn.Module):

#     def __init__(self, num_classes=200):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

        

#     def forward(self, x):
#         x_feat = self.features(x)
#         x = self.avgpool(x_feat)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x_feat, x
    

# RESNET FOR CIFAR 10

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out1 = self.layer1(out)
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         out5 = F.avg_pool2d(out4, 4)
#         outf = out5.view(out5.size(0), -1)
#         out = self.linear(outf)
#         return out, outf, [out1, out2, out3, out4, out5]


# # Factory functions
# def ResNet18(num_classes=10):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# def ResNet50(num_classes=10):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


# # Decoder and Discriminator
# class Decoder(nn.Module):
#     def __init__(self, in_features=512):
#         super(Decoder, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_features, 256 * 2 * 2),
#             nn.ReLU(True)
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), 256, 2, 2)
#         return self.deconv(x)



# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, stride=2, padding=1),   # 32x32 -> 16x16
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16 -> 8x8
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8x8 -> 4x4
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Flatten(),  # [B, 256, 4, 4] -> [B, 4096]
#             nn.Linear(256 * 4 * 4, 1)
#         )

#     def forward(self, x):
#         return self.main(x)



# # DistillModel wrapper if needed
# class DistillModel(nn.Module):
#     def __init__(self, feature_extractor, classifier):
#         super(DistillModel, self).__init__()
#         self.feature_extractor = feature_extractor
#         self.classifier = classifier

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         out = self.classifier(features)
#         return out, features
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.models import resnet50


# # --- Residual Blocks for ResNet (Used in ResNet18/ResNet50 if training from scratch) ---
# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
        
#         # Handle size mismatch
#         shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
#         if out.size() != shortcut.size():
#             # Calculate necessary padding
#             diff_h = out.size()[2] - shortcut.size()[2]
#             diff_w = out.size()[3] - shortcut.size()[3]
#             shortcut = F.pad(shortcut, (0, diff_w, 0, diff_h))
        
#         out += shortcut
#         out = F.relu(out)
#         return out


# # --- Pretrained ResNet50-based Teacher Feature Extractor ---
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = resnet50(weights=None)  # If training from scratch; else load pre-trained
#         self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and FC

#     def forward(self, x):
#         x = self.features(x)  # [B, 2048, H/32, W/32]
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.flatten(x, 1)  # [B, 2048]
#         return x


# # --- Classifier Head ---
# class Classifier(nn.Module):
#     def __init__(self, in_features=2048, num_classes=100):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(in_features, num_classes)

#     def forward(self, x):
#         return self.fc(x)


# # --- Decoder for Reconstruction ---
# # class Decoder(nn.Module):
# #     def __init__(self, in_features=2048):
# #         super(Decoder, self).__init__()
# #         self.fc = nn.Sequential(
# #             nn.Linear(in_features, 256 * 2 * 2),
# #             nn.ReLU(True)
# #         )
# #         self.deconv = nn.Sequential(
# #             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
# #             nn.ReLU(True),
# #             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
# #             nn.ReLU(True),
# #             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
# #             nn.ReLU(True),
# #             nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
# #             nn.Tanh()
# #         )

# #     def forward(self, x):
# #         x = self.fc(x)
# #         x = x.view(x.size(0), 256, 2, 2)
# #         return self.deconv(x)
# class Decoder(nn.Module):
#     def __init__(self, in_features=512):  # Changed default to 512
#         super(Decoder, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(in_features, 256 * 2 * 2),
#             nn.ReLU(True)
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.view(x.size(0), 256, 2, 2)
#         return self.deconv(x)


# # --- Discriminator ---
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()
# #         self.main = nn.Sequential(
# #             nn.Conv2d(3, 64, 4, stride=2, padding=1),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Conv2d(64, 128, 4, stride=2, padding=1),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Conv2d(128, 256, 4, stride=2, padding=1),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Flatten(),
# #             nn.Linear(256 * 4 * 4, 1)
# #         )

# #     def forward(self, x):
# #         return self.main(x)

# # models.py
# class Discriminator(nn.Module):
#     def __init__(self, in_dim=512, hidden=256):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, 1),
#             nn.Sigmoid()  # Optional depending on loss usage
#         )

#     def forward(self, x):
#         return self.main(x)

# # --- Distill Wrapper Model ---
# class DistillModel(nn.Module):
#     def __init__(self, feature_extractor, classifier):
#         super(DistillModel, self).__init__()
#         self.feature_extractor = feature_extractor
#         self.classifier = classifier

#     def forward(self, x):
#         features = self.feature_extractor(x)
#         logits = self.classifier(features)
#         return logits, features


# # Optional ResNet complete (used only if not using torchvision) - Unused in typical pretrained setup
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out1 = self.layer1(out)
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
#         out4 = self.layer4(out3)
#         out5 = F.avg_pool2d(out4, 4)
#         outf = out5.view(out5.size(0), -1)
#         out = self.linear(outf)
#         return out, outf, [out1, out2, out3, out4, out5]


# def ResNet18(num_classes=10):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# def ResNet50(num_classes=10):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.profiler import profile, ProfilerActivity

# # ---------------------- BasicBlock & Bottleneck ----------------------
# class Bottleneck(nn.Module):
#     expansion = 4
#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*planes)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         return F.relu(out)

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         return F.relu(out)

# # ------------------------- ResNet -------------------------
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=100):
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNet18(num_classes=100):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

# def ResNet50(num_classes=100):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# # ------------------------ Utilities ------------------------
# def count_parameters(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable

# def benchmark_inference_time(model, input_shape=(1, 3, 32, 32), device=torch.device("cpu")):
#     model = model.to(device).eval()
#     input_tensor = torch.randn(input_shape).to(device)
#     with torch.no_grad():
#         for _ in range(10):  # Warm-up
#             _ = model(input_tensor)
#     with profile(activities=[ProfilerActivity.CPU]) as prof:
#         with torch.no_grad():
#             _ = model(input_tensor)
#     total_time_ms = sum([evt.self_cpu_time_total for evt in prof.key_averages()])
#     return total_time_ms / 1000  # convert µs to ms

# # ------------------------ Main ------------------------
# if __name__ == "__main__":
#     device = torch.device("cpu")

#     print("ResNet18")
#     net18 = ResNet18()
#     p18_total, p18_trainable = count_parameters(net18)
#     t18 = benchmark_inference_time(net18, device=device)
#     print(f"Total Parameters     : {p18_total}")
#     print(f"Trainable Parameters : {p18_trainable}")
#     print(f"Inference Time (CPU) : {t18:.2f} ms\n")

#     print("ResNet50")
#     net50 = ResNet50()
#     p50_total, p50_trainable = count_parameters(net50)
#     t50 = benchmark_inference_time(net50, device=device)
#     print(f"Total Parameters     : {p50_total}")
#     print(f"Trainable Parameters : {p50_trainable}")
#     print(f"Inference Time (CPU) : {t50:.2f} ms")
