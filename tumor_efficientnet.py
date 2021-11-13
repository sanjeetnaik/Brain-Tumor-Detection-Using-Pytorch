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
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)

train_loader = dataloaders['train']
test_loader = dataloaders['val']

inputs, classes = next(iter(dataloaders['train']))

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b4')

# Unfreeze model weights
for param in model.parameters():
    param.requires_grad = True

num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(class_names))

optimizer = optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

# Train model
loss_log = []

for epoch in range(2):    
    model.train()    
    for ii, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        # target = target.float()                

        optimizer.zero_grad()
        output = model(data)                
    
        loss = loss_func(output, target)
        loss.backward()

        optimizer.step()  
        
        if ii % 1000 == 0:
            loss_log.append(loss.item())
       
    print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))

predict = []
model.eval()
correct = 0
total = 0
for i, (data, labels) in enumerate(test_loader):
    # data = data.cuda()
    output = model(data)    

    pred = output
    predicted_vals = pred > 0.5
    # predict.append(int(predicted_vals))

    _, preds = torch.max(output, 1)

    correct += torch.sum(preds == labels.data)
    
print(f'Accuracy : {correct/len(test_loader)*100}')