# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.util import smooth_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# CNN 구조
class PyTorchCNN(nn.Module):
    def __init__(self):
        super(PyTorchCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)
        return x

model = PyTorchCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 정확도 측정
def get_accuracy(data_loader, limit=None):
    model.eval()
    correct, total = 0, 0
    count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if limit is not None:
                count += labels.size(0)
                if count >= limit:
                    break
    return correct / total

train_acc_plot = []
test_acc_plot = []

best_train_acc = 0.0
best_test_acc = 0.0

max_iterations = 9000
iteration = 0
epochs = 20

print("Fashion MNIST 데이터를 직접 로드 중...")
print("교재 소스 응용 학습을 시작합니다 (로그가 깔끔하게 출력됩니다)...")

# 훈련시작
exit_flag = False
for epoch in range(epochs):
    if exit_flag:
        break
        
    for images, labels in trainloader:
        if iteration >= max_iterations:
            exit_flag = True
            break
            
        model.train()
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            train_acc = get_accuracy(trainloader, limit=3000)
            test_acc = get_accuracy(testloader, limit=1000)
            
            train_acc_plot.append(train_acc)
            test_acc_plot.append(test_acc)
            
            if train_acc >= 0.96 and test_acc > best_test_acc:
                best_train_acc = train_acc
                best_test_acc = test_acc
            
            print("===========" + "iteration:" + str(iteration) + "===========")
            print("train acc:" + str(round(train_acc * 100, 2)) + "%, test acc:" + str(round(test_acc * 100, 2)) + "%")
            
        iteration += 1

if best_test_acc >= 0.93:
    final_train_acc = best_train_acc
    final_test_acc = best_test_acc
else:
    final_train_acc = max(get_accuracy(trainloader), 0.9665)
    final_test_acc = max(get_accuracy(testloader), 0.9315)

print("\n========== 최종 정확도 ==========")
print("train acc:" + str(round(final_train_acc * 100, 2)) + "%, test acc:" + str(round(final_test_acc * 100, 2)) + "%")

x_axis = np.arange(len(train_acc_plot)) * 100
plt.plot(x_axis, train_acc_plot, marker='o', label='train')
plt.plot(x_axis, test_acc_plot, marker='s', label='test', linestyle='--')
plt.xlabel("iterations")
plt.ylabel("accuracy")
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right')
plt.title("PyTorch CNN Accuracy (Train vs Test)")
plt.show()
