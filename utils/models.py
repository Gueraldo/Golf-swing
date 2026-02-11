import torch.nn as nn
from torchvision.models import resnet34

class BaselineModel(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        
        resnet = resnet34()
        self.backbone = nn.Sequential(
            resnet.conv1,   # 64 channels
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2,  # 128 channels
            resnet.layer3,  # 256 channels
            resnet.layer4   # 512 channels
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, num_keypoints, kernel_size=1, stride=1),
            # nn.Sigmoid()
        )



    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)

        return x