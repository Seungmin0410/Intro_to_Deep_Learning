import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import time
import torch.nn.utils.prune as prune

trainset = torchvision.datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

testset = torchvision.datasets.MNIST(
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
            ('c1', nn.Conv2d(1, 6, kernel_size=(5,5))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2,2), stride=2)),
            ('c2', nn.Conv2d(6, 16, kernel_size=(3,3))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2,2), stride=2)),
            ('c3', nn.Conv2d(16, 120, kernel_size=(5,5))),
            ('relu3', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(120,84)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(84,10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def train(model, trainloader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels) 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.3f}, Train Accuracy: {100 * correct / total:.2f}%')

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def main():
    model = LeNet5()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

    for epoch in range(15):
        train(model, trainloader, optimizer, criterion, epoch)
    '''
    print(f'No Pruning: Test Accuracy: {test(model, testloader):.2f}%')
    '''
    prune.l1_unstructured(model.convnet.c1, name = 'weight', amount = 0.7)
    prune.l1_unstructured(model.convnet.c2, name = 'weight', amount = 0.7)
    prune.l1_unstructured(model.convnet.c3, name = 'weight', amount = 0.7)

    print(f'Pruning 70%: Test Accuracy: {test(model, testloader):.2f}%')

    for epoch in range(5):
        train(model, trainloader, optimizer, criterion, epoch)

    print(f'After Fine-tuning: Test Accuracy: {test(model, testloader):.2f}%')
    
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight
            total += weight.numel()
            zeros += (weight == 0).sum().item()
    print(f'Total Parameter: {total}')
    print(f'Pruned Parameter: {zeros}')
    print(f'Pruned Ratio: {100 * zeros / total:.2f}%')

if __name__ == '__main__':
    main()


