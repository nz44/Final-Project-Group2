##########################################
#                                                                                                                        #
#                                                                                                                        #
#        I  Create Train and Test Data Folders                                                  #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
# #########################################

# You should have the kaggle dataset already downloaded and unzipped, as
# described in the readme file.
# https://stackabuse.com/creating-and-deleting-directories-with-python/
import os
base_path = os.getcwd()
print ("The current working directory is %s" % base_path)

folder_name_level_1 = ['train_data', 'test_data']
folder_name_level_2 = ['Vincent_van_Gogh', 'Pablo_Picasso',\
               'Francisco_Goya', 'Alfred_Sisley', 'Leonardo_da_Vinci', 'Edgar_Degas', \
               'Rembrandt', 'Andy_Warhol', 'Paul_Gauguin', 'Salvador_Dali']

# Create empty folders for training and testing data, with classes as sub-folders.
for name in folder_name_level_1:
    if not os.path.exists(os.path.join(base_path, name)):
        os.mkdir(os.path.join(base_path, name))
    else:
        pass

for name_1 in folder_name_level_1:
    for name_2 in folder_name_level_2:
        if not os.path.exists(os.path.join(base_path, name_1, name_2)):
            os.mkdir(os.path.join(base_path, name_1, name_2))
        else:
            pass

train_split = {'Vincent_van_Gogh':702, 'Edgar_Degas':562, 'Pablo_Picasso':351, \
                    'Paul_Gauguin':249, 'Francisco_Goya':233, 'Rembrandt':210, 'Alfred_Sisley':207,\
                    'Andy_Warhol':145, 'Leonardo_da_Vinci':114, 'Salvador_Dali':111}


# Move files to  the training set folders
for key, val in train_split.items():
    for i in range(1, val+1):
        if not os.path.isfile(os.path.join(base_path, 'train_data', key , key+'_'+str(i)+'.jpg')):
            os.rename(os.path.join(base_path, 'resized', key+ '_'+str(i) +'.jpg'), os.path.join(base_path, 'train_data', key , key+'_'+str(i)+'.jpg'))
        else:
            pass

test_split = {'Vincent_van_Gogh':[702, 877], 'Edgar_Degas':[562, 702], 'Pablo_Picasso':[351, 439],\
                    'Paul_Gauguin':[249, 311], 'Francisco_Goya':[233, 291], 'Rembrandt':[210, 262], 'Alfred_Sisley':[207, 259],\
                    'Andy_Warhol':[145, 181], 'Leonardo_da_Vinci':[114, 143], 'Salvador_Dali':[111, 139]}


# Move files to  the testing set folders
for key, val in test_split.items():
    for i in range(val[0]+1, val[1]+1):
        if not os.path.isfile(os.path.join(base_path, 'test_data', key , key+'_'+str(i)+'.jpg')):
            os.rename(os.path.join(base_path, 'resized', key+ '_'+str(i) +'.jpg'), os.path.join(base_path, 'test_data', key , key+'_'+str(i)+'.jpg'))
        else:
            pass




##########################################
#                                                                                                                        #
#                                                                                                                        #
#        II  Load Data and Preprocess                                                                #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
#                                                                                                                        #
# #########################################

# Define the transforms needed to apply to images
import torchvision.transforms as transforms
# https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
transformations = transforms.Compose([
    transforms.Resize(800),
    transforms.CenterCrop(800),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
# ----------------------------------------------------------------------------------------------------------------
# Load files into pytorch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


print()
print('This is train_data class label mapping:')
train_data = ImageFolder(os.path.join(base_path, 'train_data'), transform= transformations)
print(train_data.class_to_idx)
print('The number of paintings in train dataset: ')
print(len(train_data))

print()
print('This is test_data class label mapping:')
test_data = ImageFolder(os.path.join(base_path, 'test_data'), transform=transformations)
print(test_data.class_to_idx)
print('The number of paintings in test dataset: ')
print(len(test_data))


# https://discuss.pytorch.org/t/questions-about-imagefolder/774/6
# https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/6-Conv_Mnist/Conv_Mnist_gpu.py
from torch.utils.data import DataLoader
batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


classes = ('Alfred_Sisley', 'Andy_Warhol', 'Edgar_Degas', 'Francisco_Goya', 'Leonardo_da_Vinci',
           'Pablo_Picasso', 'Paul_Gauguin', 'Rembrandt', 'Salvador_Dali', 'Vincent_van_Gogh')

# ----------------------------------------------------------------------------------------------------------------
# Take a look how the image look like
# https://github.com/amir-jafari/Deep-Learning/blob/master/Pytorch_/Mini_Project/FashionMNIST.py
import matplotlib.pyplot as plt
plt.interactive(False)
import numpy as np

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    # change to matplotlib width, height, channel from channel, width, height
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
import torchvision
imshow(torchvision.utils.make_grid(images))
plt.show()

# print labels
print()
print('Labels of randomly chosen 4 paintings:')
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


