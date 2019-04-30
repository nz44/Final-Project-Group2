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

import os
base_path = os.getcwd()
print ("The current working directory is %s" % base_path)

train_data = ImageFolder(os.path.join(base_path, 'train_data'), transform= transformations)
test_data = ImageFolder(os.path.join(base_path, 'test_data'), transform=transformations)



##########################################
#                                                                                                                        #
#                                                                                                                        #
#        CNN Model 1                                                                                           #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
# #########################################
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
learning_rate = 1e-3
num_epochs = 3

#############################
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
# Time the training
train_loss = []
num_iterations = []

mini_batches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for b in mini_batches:
    batch_size = b
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    start = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
               print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_data) // batch_size, loss.item()))

            train_loss.append(loss.item())
            num_iterations.append((i+1)*(epoch+1))

    cnn.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
          images = Variable(images).cuda()
          outputs = cnn(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted.cpu() == labels).sum()
    print('Test Accuracy of the model 1 on the ', len(test_data), ' test images: %d %%' % (100 * correct / total))

    end = time. time()
    print()
    print('The time taken to train the model 1 with mini batch size', batch_size, 'is :')
    print(end - start)
