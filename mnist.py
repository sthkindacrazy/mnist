import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(), download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=100, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(784, 50)
        self.layer2 = nn.Linear(50, 10)

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        return output

lossFunction = nn.CrossEntropyLoss()

net  = Net()
num_epochs = 5
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1,28*28)
        out = net(images)
        loss = lossFunction(out, labels)


correct = 0
total = 0
for images,labels in test_loader:
    images = images.reshape(-1,28*28)
    out = net(images)
    _,predicted = torch.max(out.data,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))




