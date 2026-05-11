import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import time

trainset = torchvision.datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

testset = torchvision.datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False
)

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(5,5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2,2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5,5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2,2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5,5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120,84)),
            ('relu6', nn.ReLU()),
            ('drop', nn.Dropout(p=0.2)),
            ('f7', nn.Linear(84,10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))
    
    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
model = LeNet5()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

start_time = time.time()

for epoch in range(15):
    correct_train = 0
    total_train = 0
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}, Train Accuracy: {100 * correct_train / total_train:.2f}%')

end_time = time.time()
print(f'Training Time: {end_time - start_time:.2f} seconds')

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        output = model(images)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')