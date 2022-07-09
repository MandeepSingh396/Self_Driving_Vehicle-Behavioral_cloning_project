#necessary packages
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import csv

# Loading the data

lines = []
with open ('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    c = 0
    for line in reader:
        c+=1
        if(c>1):
            lines.append(line)

t = transforms.ToPILImage()
tfm = transforms.RandomHorizontalFlip(p=1.0)

x_train = []
y_train = []

for line in lines:
    for i in range(0,3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = "data/IMG" + "/" + filename
        img = image.imread(current_path)
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)
        img_1 = t(img)
        img_flip = tfm(img)
        img_flip_1 = t(img_flip)
        x_train.append(img_1)
        x_train.append(img_flip_1)
        measurement = float(line[i+3])
        y_train.append(measurement)
        y_train.append((-1)*measurement)

X_train = np.array(x_train)
Y_train = np.array(y_train)
Y_train = np.reshape(Y_train, (-1,1))

# Modifying the dataset class to input images and labels separately

class YourDataset(Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):

        x = self.x_train[idx]
        y = self.y_train[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# DataLoader generation

from torchvision.transforms.functional import crop

def cropimg(image):    
    return crop(image,70,0,65,320)         #Image crop (top 70 pixels and bottom 25 pixels cropped)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(cropimg),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = YourDataset(X_train, Y_train, transform = transform)
batch_size = 128
train_ds, val_ds = random_split(trainset, [57372, 15000])

train_loader = DataLoader(train_ds, batch_size, shuffle = True)
val_loader = DataLoader(val_ds, batch_size)

# data visualization

def imshow(img):
    img = img/2 + 0.5  #unnormalize
    plt.imshow(np.transpose(img, (1,2,0))) #convert from tensor image
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

fig = plt.figure(figsize = (25,4))

for idx in np.arange(40):
    ax = fig.add_subplot(4, int(20/2), idx+1, xticks=[], yticks = [])
    imshow(images[idx])


# Model Architecture - Nvidia_Dave 2 

import torch.nn as nn
import torch.nn.functional as F

class Nvidia(nn.Module):
    def __init__(self):
        super(Nvidia, self).__init__()
        
        self.conv1 = nn.Conv2d(3,24,5, padding = 0, stride=2) #in-channels, out-channels, kernel_size
        self.conv2 = nn.Conv2d(24,36,5, padding = 0, stride=2)
        self.conv3 = nn.Conv2d(36,48,5, padding = 0, stride=2)
        self.conv4 = nn.Conv2d(48,64,3, padding = 0)
        self.conv5 = nn.Conv2d(64,64,3, padding = 0)
        self.fc1 = nn.Linear(64*1*33, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 64*1*33)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x 
    
model = Nvidia()
model

#Training using GPU

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available')
else:
    print('CUDA is available')

if train_on_gpu:
    model.cuda()

# Training

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

n_epochs = 2

val_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    train_loss = 0
    val_loss = 0
    
    for data, target in train_loader:
        data = data.float()
        target = target.float()
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion (output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        
# Validation

    model.eval()  
    for data, target in val_loader:
        data = data.float()
        target = target.float()
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion (output, target)
        val_loss += loss.item()*data.size(0)
        
    train_loss = train_loss/len(train_loader.dataset)
    val_loss = val_loss/len(val_loader.dataset)
    
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))
    
    if val_loss <= val_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).    Saving model ...'.format(val_loss_min, val_loss))
        torch.save(model.state_dict(), 'model_Nvidia.h5')
        val_loss_min = val_loss