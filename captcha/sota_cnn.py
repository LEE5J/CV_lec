import numpy as np
import timm
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models


class sota(nn.Module):
    def __init__(self, num_classes=10):
        super(sota, self).__init__()
        # 사전 훈련된 ResNext50 모델 불러오기
        self.resnext50                                                                                                                               = models.resnext50_32x4d()
        # 마지막 레이어 수정
        self.resnext50.fc = nn.Linear(self.resnext50.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnext50(x)
        return x


def forward(self, x):
        return self.efficientnet(x)


if __name__ == '__main__':
    # 모델 인스턴스화
    model = sota(num_classes=10)