import torch
import torchvision
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import torch.nn.functional as F 
from sklearn.utils import shuffle
import torch.optim as optim
import cv2
from tqdm import tqdm

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    transforms.Resize((150,150))
    ])

data_dir = 'data/brain_tumors'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          transform)
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

train_loader = dataloaders['train']
test_loader = dataloaders['val']



class ConvNet(nn.Module):
    def __init__(self,num_classes=4):
        super(ConvNet,self).__init__()
        
         
        self.conv1=nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        
        self.bn1=nn.BatchNorm2d(num_features=12)
        self.relu1=nn.ReLU()        
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
        self.relu2=nn.ReLU()
        self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=32)
        self.relu3=nn.ReLU()
        self.fc=nn.Linear(in_features=75 * 75 * 32,out_features=num_classes)
        
        
        
        #Feed forwad function
        
    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output=self.relu1(output)
        output=self.pool(output)
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.conv3(output)
        output=self.bn3(output)
        output=self.relu3(output)            
        output=output.view(-1,32*75*75)
        output=self.fc(output)
            
        return output
    
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
model=ConvNet(num_classes=4).to(device)

#Optmizer and loss function
optimizer=optim.Adam(model.parameters(),lr=0.001)
criterion=nn.CrossEntropyLoss()



best_accuracy=0.0

# for epoch in range(20):
    
#     #Evaluation and training on training dataset
#     model.train()
#     train_accuracy=0.0
#     train_loss=0.0
    
#     for i, (images,labels) in enumerate(train_loader):            
#         optimizer.zero_grad()
#         outputs=model(images)
#         loss=loss_function(outputs,labels)
#         loss.backward()
#         optimizer.step()
        
        
#         train_loss+= loss.cpu().data*images.size(0)
#         _,prediction=torch.max(outputs.data,1)
        
#         train_accuracy+=int(torch.sum(prediction==labels.data))
        
#     train_accuracy=train_accuracy/len(train_loader)
#     train_loss=train_loss/len(train_loader)
    
    
#     # Evaluation on testing dataset
#     model.eval()
    
#     test_accuracy=0.0
#     for i, (images,labels) in enumerate(test_loader):            
#         outputs=model(images)
#         _,prediction=torch.max(outputs.data,1)
#         test_accuracy+=int(torch.sum(prediction==labels.data))
    
#     test_accuracy=test_accuracy/len(test_loader)
    
    
#     print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
#     if test_accuracy>best_accuracy:
#         torch.save(model.state_dict(),'best_checkpoint.model')
#         best_accuracy=test_accuracy

n_total_steps = len(train_loader)
num_epochs = 1
print('hi')
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        # for i in range(4):
        #     label = labels[i]
        #     pred = predicted[i]
        #     if (label == pred):
        #         n_class_correct[label] += 1
        #     n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    # for i in range(10):
    #     acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    #     print(f'Accuracy of {class_names[i]}: {acc} %')

# torch.save(model.state_dict(), r'C:\Users\sanje\Desktop\Jimmy')
PATH = r'C:\Users\sanje\pytorch'
torch.save(model.state_dict(), os.path.join(PATH,'model2.pth'))