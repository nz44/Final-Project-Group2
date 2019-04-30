import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision
# https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
transformations = transforms.Compose([
    transforms.Resize(800),
    transforms.CenterCrop(800),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_data = ImageFolder(root='/home/ubuntu/Deep-Learning/Naixin_Final_Pytorch_CNN/train_data/', transform= transformations)
test_data = ImageFolder(root='/home/ubuntu/Deep-Learning/Naixin_Final_Pytorch_CNN/test_data/', transform=transformations)

batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# CNN Model  1
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=10, stride=5,  padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(4))
        self.layer2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=10, stride=5,  padding=2),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(4))
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
# -----------------------------------------------------------------------------------
cnn = CNN()
cnn.cuda()
# -----------------------------------------------------------------------------------
# Hyperparameters
learning_rates = [1, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
num_epochs = 1

#############################
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()

for learning_rate in learning_rates:
       optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
       start = time. time()
       for epoch in range(num_epochs):
             for i, (images, labels) in enumerate(train_loader):
                  images = Variable(images).cuda()
                  labels = Variable(labels).cuda()
                  optimizer.zero_grad()
                  outputs = cnn(images)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()

                  if (i + 1) % 100 == 0:
                         print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                                  % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))
       cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
       correct = 0
       total = 0
       for images, labels in test_loader:
             images = Variable(images).cuda()
             outputs = cnn(images)
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted.cpu() == labels).sum()
            # -----------------------------------------------------------------------------------
       print('Test Accuracy of the model 1 on the ', len(test_data),
                  ' test images: %d %%' % (100 * correct / total), 'with learning rate', learning_rate)
       end = time. time()
       print()
       print('The time taken to train the model 1 with learning rate', learning_rate, 'is: ')
       print(end - start)

