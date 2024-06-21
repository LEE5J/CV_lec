import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
# from captcha.sota_cnn import sota  # 65~63% 0.001에 60에폭
import torch.nn.functional as F

from captcha.utility import load_images_from_folder, transform_test, process_and_load_images, load_target_image, \
    process_and_load_targets, print_t1_result

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SplitAttention(nn.Module):
    def __init__(self, channels, radix, reduction_factor=4):
        """
        channels: 입력 채널 수
        radix: 분할할 그룹 수
        reduction_factor: Reduction ratio for the attention network
        """
        super(SplitAttention, self).__init__()
        self.radix = radix
        self.channels = channels
        inter_channels = max(channels // reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels//radix, inter_channels, 1, groups=radix)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels, 1, groups=radix)

        self.rsoftmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channels, height, width = x.size()
        split = torch.split(x, self.channels // self.radix, dim=1)
        gap = sum([item.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True) for item in split])

        gap = F.relu(self.bn1(self.fc1(gap)))
        atten = self.fc2(gap)
        atten = atten.view(batch, self.radix, self.channels // self.radix, 1, 1)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        out = [atten[:, i::self.radix, :, :] * split[i] for i in range(self.radix)]
        out=torch.cat(out, dim=1)
        return out.contiguous()


# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.attention1 = SplitAttention(32, radix=4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.attention2 = SplitAttention(64, radix=4)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.attention3 = SplitAttention(128, radix=4)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.attention4 = SplitAttention(128, radix=4)
        self.fc = nn.Sequential(
            nn.Linear(8*8 * 128, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 20),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        # layer1을 통과한 후 attention 적용
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.layer3(x)
        x = self.attention3(x)
        x = self.layer4(x)
        x = self.attention4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def feature(self,x):
        x = self.layer1(x)
        x = self.attention1(x)
        x = self.layer2(x)
        x = self.attention2(x)
        x = self.layer3(x)
        x = self.attention3(x)
        x = self.layer4(x)
        x = self.attention4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        return x



def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')
        if total_loss / len(train_loader) <0.02:
            break

def eval(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy





# 데이터 로딩 및 분리
def load_and_split_data( test_size=0.2, random_state=42):
    images, labels = load_images_from_folder('data')
    # 학습 데이터와 테스트 데이터로 분리
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=test_size, random_state=random_state
    )
    # 이미지 및 레이블 처리
    train_images, train_labels = process_and_load_images(train_images, train_labels)
    test_images, test_labels = process_and_load_images(test_images, test_labels)
    return train_images, train_labels, test_images, test_labels

def main():
    train_images, train_labels, test_images, test_labels = load_and_split_data()
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 학습 및 평가 실행
    model = CNN().to(device)  # renext 기준 70프로 데이터 셋이 더 필요할지도
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train(model, train_loader, criterion, optimizer, epochs=100)  # 100:64,50:61
    if eval(model, test_loader)>68:
        return model

mode = "test"
if mode == "save" and __name__ == "__main__":
    i=0
    while(True):
        if i==3:
            break
        model = main()
        if model is not None:
            torch.save(model.state_dict(),f'cnn2{i}.pth')
            i+=1

if mode == "test" and __name__ == "__main__":
    images, filenames = load_target_image('target')
    images = process_and_load_targets(images)
    for i in range(3):
        model = CNN().to(device)
        state_dict = torch.load(f"cnn{i}.pth")
        model.load_state_dict(state_dict)
        test_dataset = TensorDataset(images)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for batch_images in test_loader:
                batch_images = batch_images[0].to(device)
                outputs = model(batch_images)
                _, predicted = torch.max(outputs.data, 1)
                preds.extend(predicted)
        print_t1_result(filenames,preds,challenge="c2",attmp=i)


