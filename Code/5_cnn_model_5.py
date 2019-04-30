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

batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
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
        self.dropout = nn.Dropout2d(p=0.5)
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
num_epochs = 20

#############################
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# -----------------------------------------------------------------------------------
# Train the Model
# Time the training
start = time. time()
train_loss = []
num_iterations = []
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

end = time. time()
print()
print('The time taken to train the model 5 is: ')
print(end - start)
# -----------------------------------------------------------------------------------
# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
y_true = []
y_pred = []
for images, labels in test_loader:
    y_true.append(labels.tolist())
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    y_pred.append(predicted.cpu().tolist())
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()
# -----------------------------------------------------------------------------------
print('Test Accuracy of the model 1 on the ', len(test_data), ' test images: %d %%' % (100 * correct / total))
# -----------------------------------------------------------------------------------
# Plot Confusion Matrix


flat_y_true = []
for sublist in y_true:
    for item in sublist:
        flat_y_true.append(item)

flat_y_pred = []
for sublist in y_pred:
    for item in sublist:
        flat_y_pred.append(item)

classes = ('Alfred_Sisley', 'Andy_Warhol', 'Edgar_Degas', 'Francisco_Goya', 'Leonardo_da_Vinci',
           'Pablo_Picasso', 'Paul_Gauguin', 'Rembrandt', 'Salvador_Dali', 'Vincent_van_Gogh')

for n, i in enumerate(flat_y_true):
    if i == 0:
        flat_y_true[n] = 'Alfred_Sisley'
    if i == 1:
        flat_y_true[n] = 'Andy_Warhol'
    if i == 2:
        flat_y_true[n] = 'Edgar_Degas'
    if i == 3:
        flat_y_true[n] = 'Francisco_Goya'
    if i == 4:
        flat_y_true[n] = 'Leonardo_da_Vinci'
    if i == 5:
        flat_y_true[n] =  'Pablo_Picasso'
    if i == 6:
        flat_y_true[n] = 'Paul_Gauguin'
    if i == 7:
        flat_y_true[n] = 'Rembrandt'
    if i == 8:
        flat_y_true[n] = 'Salvador_Dali'
    if i == 9:
        flat_y_true[n] = 'Vincent_van_Gogh'

for n, i in enumerate(flat_y_pred):
    if i == 0:
        flat_y_pred[n] = 'Alfred_Sisley'
    if i == 1:
        flat_y_pred[n] = 'Andy_Warhol'
    if i == 2:
        flat_y_pred[n] = 'Edgar_Degas'
    if i == 3:
        flat_y_pred[n] = 'Francisco_Goya'
    if i == 4:
        flat_y_pred[n] = 'Leonardo_da_Vinci'
    if i == 5:
        flat_y_pred[n] =  'Pablo_Picasso'
    if i == 6:
        flat_y_pred[n] = 'Paul_Gauguin'
    if i == 7:
        flat_y_pred[n] = 'Rembrandt'
    if i == 8:
        flat_y_pred[n] = 'Salvador_Dali'
    if i == 9:
        flat_y_pred[n] = 'Vincent_van_Gogh'

# print('This is test dataset true labels:')
# print(flat_y_true)
# print()
# print('This is test dataset predicted labels: ')
# print(flat_y_pred)

cm = confusion_matrix(flat_y_true, flat_y_pred, labels = classes)
print()
print('This is confusion matrix: ')
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Model 5 Classifier')
fig.colorbar(cax)
ax.set_yticks([1,2,3,4,5,6,7,8,9,10])
ax.set_yticklabels(classes, rotation=40)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# -----------------------------------------------------------------------------------
# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn_model_5.pkl')

##########################################
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
#             Check a random batch's prediction with its actual labels                #                                                                                                      #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
# #########################################


plt.interactive(False)

dataiter = iter(test_loader)
images, labels = dataiter.next()

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    # change to matplotlib width, height, channel from channel, width, height
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


imshow(torchvision.utils.make_grid(images))
plt.show()

# print labels
print()
print('Randomly select a batch and print their labels: ')
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

print()
print('The predicted class labels for above randomly selected batch: ')
images = Variable(images).cuda()
outputs = cnn(images)
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


##########################################
#                                                                                                                        #
#                                                                                                                        #
#        Plot Train Loss with Iterations                                                              #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
# #########################################


# print(num_iterations)
# print(train_loss)
plt.plot(num_iterations, train_loss)
plt.title('Figure 5. The Train Loss for Model 5')
plt.show()
