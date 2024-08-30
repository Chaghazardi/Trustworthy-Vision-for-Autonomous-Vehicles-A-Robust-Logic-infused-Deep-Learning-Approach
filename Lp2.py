
import time
import math
import os
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import random
import shutil
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import numpy as np
import pickle
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data as data_utils

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(42)

if torch.cuda.is_available():
    print("GPU available!")
else:
    print("GPU not available.")

batch_size=32
LOGIC = True


def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, _, _, _, class_targets in dataloader:
            class_preds = model(images)
            
            _, class_predicted = torch.max(class_preds, 1)
            total += class_targets.size(0)
            correct += (class_predicted == class_targets).sum().item()
    return correct / total




# Plot Loss (Train and Validation) in one subplot
def plot_training_validation( history_1, history_3, filename=None):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)

    plt.plot(history_1['train_loss_1'], label='Model 1 (Train Loss)', linestyle='dashed', color='blue')
    plt.plot(history_3['train_loss_3'], label='Model 3 (Train Loss)', linestyle='dashed', color='red')

    plt.plot(history_1['validation_loss_1'], label='Model 1 (Validation Loss)', linestyle='dashdot', color='blue')
    plt.plot(history_3['validation_loss_3'], label='Model 3 (Validation Loss)', linestyle='dashdot', color='red')

    plt.title('Training and Validation Loss_ Lp2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)

    plt.plot(history_1['train_acc_1'], label='Model 1 (Train Accuracy)', color='blue')
    plt.plot(history_3['train_acc_3'], label='Model 3 (Train Accuracy)', color='red')
    
    plt.plot(history_1['validation_acc_1'], label='Model 1 (Validation Accuracy)', linestyle='dotted', color='blue')
    plt.plot(history_3['validation_acc_3'], label='Model 3 (Validation Accuracy)', linestyle='dotted', color='red')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    path = os.path.join(figure_folder, filename)
    plt.savefig(path)
    plt.show()


# Label Overview
classes = { 5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)' , 
            9:'No passing',  
            12:'Priority road', 
            13:'Yield', 
            14:'Stop',  
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            41:'End of no passing' }


colours = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'black'}
Shapes  = {0: 'triangle', 1: 'circle', 2: 'diamond', 3: 'octagon', 4: 'inverse triangle'}

# class_num  = len(classes)
class_num  = 43

shape_num  = len(Shapes)
colour_num = len(colours)

class_indices = list(classes.keys())
class_names   = list(classes.values())

colour_indices = list(colours.keys())
colour_names   = list(colours.values())

shape_indices = list(Shapes.keys())
shape_names   = list(Shapes.values())



# rules = np.array([
#     [9, 1, 0],
#     [12, 2, 2],
#     [13, 4, 0],
#     [14, 3, 0],
#     [19, 0, 0],
#     [34, 1, 1],
#     [41, 1, 3]
# ])

rules = np.array([
    [5,  1, 0],   
    [6,  1, 3],
    [9,  1, 0],
    [12, 2, 2],
    [13, 4, 0],
    [14, 3, 0],
    [19, 0, 0],
    [20, 0, 0],
    [33, 1, 1],
    [34, 1, 1],
    [41, 1, 3]
])


num_rules = len(rules)
for i in range(num_rules):
    class_index = rules[i][0]
    shape_index = rules[i][1]
    color_index = rules[i][2]

Logics =['product', 'Godel']
# Logics =['product', 'Godel', 'Lukasiewicz']

def logical_loss(shape_preds, color_preds, class_preds, rules, Logic):
    num_rules = len(rules)
    product = []
    satisfaction = []
    for i in range(num_rules):
        class_index, shape_index, color_index = rules[i]

        class_preds_s = F.softmax(class_preds, dim=1)
        shape_preds_s = F.softmax(shape_preds, dim=1)
        color_preds_s = F.softmax(color_preds, dim=1)
        # the constraints containing label k are each multiplied by the value of the prediction for label k

        class_rule = class_preds_s[:, class_index]
        shape_rule = shape_preds_s[:, shape_index]
        color_rule = color_preds_s[:, color_index]

        if Logic == 'product':
            product =shape_rule * color_rule * class_rule
            satisfaction.append(torch.mean(product).item())

        elif Logic == 'Godel':
            product = torch.minimum(torch.minimum(shape_rule, color_rule), class_rule)
            satisfaction.append(torch.mean(torch.tensor(product)).item())

        # elif  Logic =='Lukasiewicz':
        #     product = (shape_rule + color_rule + class_rule - 2)
        #     satisfaction.append(torch.mean(product).item())
        

    satisfaction = max(satisfaction)
    
    return 1-satisfaction


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        # img_path = os.path.join(self.root_dir, "Deeplearning", self.annotations.iloc[idx, 0])

        image = read_image(img_path)  # Assuming you have a function read_image to load images
        shape_id = int(self.annotations.iloc[idx, 2])
        color_id = int(self.annotations.iloc[idx, 3])
        symbol_id = int(self.annotations.iloc[idx, 4])
        class_id = int(self.annotations.iloc[idx, 1])  # Added Class ID

        if self.transform:
            image = self.transform(image)

        return image, shape_id, color_id, symbol_id, class_id  # Include Class ID
        # return image, class_id  # 




# Define data transformation pipeline
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]
# (image,ang_range,shear_range,trans_range):

# num_fc_layers = 10016
# img_resize = 32 
num_fc_layers = 47904
img_resize = 64 


# data_transform = T.Compose([
#     T.Resize((img_resize, img_resize)),
#     T.ToPILImage(),
#     # T.RandomResizedCrop(size=256),
#     T.RandomRotation(degrees=15),
#     T.RandomHorizontalFlip(),
#     T.ToTensor(),
#     T.Normalize(mean_nums, std_nums)
# ])


data_transform = {'train': T.Compose([
  # T.RandomResizedCrop(size=img_resize), 
  T.Resize((img_resize, img_resize)),
  T.ToPILImage(),
  T.RandomRotation(degrees=15),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]), 'val': T.Compose([
  T.Resize((img_resize, img_resize)),
  # T.CenterCrop(size=img_resize-2), 
  T.ToPILImage(),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]), 'test': T.Compose([
  T.Resize((img_resize, img_resize)),
  T.ToPILImage(),
  # T.CenterCrop(size=img_resize-2),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]),
}



# Define the root folder containing subfolders (each with its own CSV file)
root_folder = 'datasets/train2'
test_folder = 'datasets/test2'
save_dir = 'Lp2train2Oldnum'

# Lp2AllitrtrainNew2:     train_loss =(shape_loss + color_loss + class_loss) *(1+logic_loss)
# Lp2AllitrtrainNew:      train_loss =shape_loss + (color_loss + class_loss) *(logic_loss)



# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
# List subfolders (categories) in the root folder
subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
print(subfolders)




# List subfolders (categories) in the root folder
subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
# Loop through each subfolder (category)
data_loaders = {}

for subfolder in subfolders:
    print(subfolder)
    subfolder_path = os.path.join(root_folder, subfolder)

    # Load the respective CSV files for this subfolder
    train_csv = os.path.join(subfolder_path, 'train_dataset.csv') 
    validation_csv = os.path.join(subfolder_path, 'validation_dataset.csv')
    test_csv = os.path.join(test_folder, 'test.csv')

    
    train_dataset = CustomDataset(csv_file      = train_csv, root_dir='', transform=data_transform['train'])
    validation_dataset = CustomDataset(csv_file = validation_csv, root_dir='', transform=data_transform['val'])
    test_dataset = CustomDataset(csv_file       = test_csv, root_dir='', transform=data_transform['test'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    data_loaders[subfolder] = {
    'train': train_dataloader,
    'validation': validation_dataloader,
    'test': test_dataloader
    }

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()


# Define a custom model
# image, shape_id, color_id, symbol_id, class_id
in_channels = 3
num_filters = [3, 32, 32, 64, 64, 128, 128]
fc_size1 = 1024
fc_size2 = 1024
# num_fc_layers = 10016



class CustomModel_1(nn.Module):
    def __init__(self, num_class_classes):
        super(CustomModel_1, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=1)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.dropout = nn.Dropout(p=0.5)

        # Placeholder for fully connected layers
        self.fc_layer1 = nn.Linear(num_fc_layers, fc_size1)
        self.fc_layer2 = nn.Linear(fc_size1, fc_size2)
        self.fc_layer3 = nn.Linear(fc_size2, num_class_classes)
        # self.fc_layer1 = nn.Linear(num_fc_layers, fc_size1)



    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        xp2 = self.pool2(x2)
        xd2 = self.dropout(xp2)

        x3 = self.conv3(xd2)
        x4 = self.conv4(x3)
        xp4 = self.pool4(x4)
        xd4 = self.dropout(xp4)

        x5 = F.relu(self.conv5(xd4))
        x6 = F.relu(self.conv6(x5))
        xp6 = self.pool6(x6)
        xd6 = self.dropout(xp6)

        # Flatten the outputs of convolutional layers
        layer_flat2 = torch.flatten(xd2, start_dim=1)
        layer_flat4 = torch.flatten(xd4, start_dim=1)
        layer_flat6 = torch.flatten(xd6, start_dim=1)

        # Concatenate flattened layers
        layer_flat = torch.cat((layer_flat2, layer_flat4, layer_flat6), dim=1)
        num_fc_layers = layer_flat.size(1)

        # Fully connected layers
        x = F.relu(self.dropout(self.fc_layer1(layer_flat)))
        x = F.relu(self.dropout(self.fc_layer2(x)))
        x = self.fc_layer3(x)

        return x
    
class CustomModel_3(nn.Module):
    def __init__(self, num_shape_classes, num_color_classes, num_class_classes):

        super(CustomModel_3, self).__init__()

        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=1)

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.dropout = nn.Dropout(p=0.5)

        # Placeholder for fully connected layers
        self.fc_layer1 = nn.Linear(num_fc_layers, fc_size1)
        self.fc_layer2 = nn.Linear(fc_size1, fc_size2)       
        
        # Shape classifier
        self.shape_classifier = nn.Linear(fc_size2, num_shape_classes)

        # Color classifier
        self.color_classifier = nn.Linear(fc_size2, num_color_classes)

        # Class classifier
        self.class_classifier = nn.Linear(fc_size2, num_class_classes)


    def forward(self, x):
        x0 = self.conv0(x)

        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        xp2 = self.pool2(x2)
        xd2 = self.dropout(xp2)

        x3 = self.conv3(xd2)
        x4 = self.conv4(x3)
        xp4 = self.pool4(x4)
        xd4 = self.dropout(xp4)

        x5 = F.relu(self.conv5(xd4))
        x6 = F.relu(self.conv6(x5))
        xp6 = self.pool6(x6)
        xd6 = self.dropout(xp6)

        # # Flatten the outputs of convolutional layers
        layer_flat2 = torch.flatten(xd2, start_dim=1)
        layer_flat4 = torch.flatten(xd4, start_dim=1)
        layer_flat6 = torch.flatten(xd6, start_dim=1)

        # Concatenate flattened layers
        layer_flat = torch.cat((layer_flat2, layer_flat4, layer_flat6), dim=1)
        
        # Update the fully connected layers with the correct input size
        num_fc_layers = layer_flat.size(1)


        # Fully connected layers
        x = F.relu(self.dropout(self.fc_layer1(layer_flat)))
        x = F.relu(self.dropout(self.fc_layer2(x)))
        
        shape_output = self.shape_classifier(x)

        # Color prediction
        color_output = self.color_classifier(x)

        # Class prediction
        class_output = self.class_classifier(x)

        return shape_output, color_output, class_output






itrs = 10
num_epochs = 20
alphas =[0, 1]
total_times = [] 

subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
subfolders = ['dataset_150']

for subfolder in subfolders:
    subfolder_name = subfolder
    train_dataloader      = data_loaders[subfolder_name]['train']
    validation_dataloader = data_loaders[subfolder_name]['validation']
    test_dataloader       = data_loaders[subfolder_name]['test']
    print(subfolder)
    for itr in range (itrs):
        for alpha in alphas:
            if alpha ==0:
                print('alpha:', alpha, 'itr', itr)
                best_model_state_dict_1  = None
                best_model_state_dict_3  = None
    
                best_accuracy_1 = 0.0
                best_accuracy_3 = 0.0
    
                history_1 = defaultdict(list)
                history_3 = defaultdict(list)
    
                shape_criterion  = nn.CrossEntropyLoss()
                color_criterion  = nn.CrossEntropyLoss()
                symbol_criterion = nn.CrossEntropyLoss()
                class_criterion  = nn.CrossEntropyLoss() 
    
                model_1  = CustomModel_1(num_class_classes=class_num)  # Include num_class_classes
                model_3  = CustomModel_3(num_shape_classes= shape_num, num_color_classes=colour_num, num_class_classes=class_num)  # Include num_class_classes
    
                optimizer_1      = optim.Adam(model_1.parameters(), lr=0.001, weight_decay=1e-5)
                scheduler_1     = lr_scheduler.StepLR(optimizer_1, step_size=7, gamma=0.1)
    
                optimizer_3      = optim.Adam(model_3.parameters(), lr=0.001, weight_decay=1e-5)
                scheduler_3     = lr_scheduler.StepLR(optimizer_3, step_size=7, gamma=0.1)
    
                start1 = time.perf_counter()
                for epoch in range(num_epochs):
                            # Training loop for the model with only class ID
                    
                    model_1.train()
                    losses = []
                    correct = 0
                    total = 0
                    for images, _, _, _,  class_targets in train_dataloader:
                        
                        optimizer_1.zero_grad()
                        class_preds = model_1(images)
                        class_loss = class_criterion(class_preds, class_targets)  # Include class loss
                        total += class_targets.size(0)
                        
                        # Calculate the total loss by summing shape, color, symbol, and class losses
                        train_loss = class_loss
                        losses.append(train_loss.item())
    
                        # Accumulate correct predictions
                        _, class_predicted = torch.max(class_preds, 1)
                        correct += torch.sum(class_predicted == class_targets).item()
                            
                        train_loss.backward()
                        optimizer_1.step()
                        optimizer_1.zero_grad()
                    
                    scheduler_1.step()
                    train_acc_1  = 100 * correct / total
                    train_loss_1 = np.mean(losses)  
                    print(f'model_1_ Epoch [{epoch + 1}/{num_epochs}] -   train_acc_1: {train_acc_1} -   train_loss_1: {train_loss_1}')
                
                
                    # Validation loop for the model with only class ID
                    model_1.eval()
                    correct = 0
                    total = 0
                    losses = []
                    with torch.no_grad():
                        for images, _, _, _, class_targets in validation_dataloader:
                            class_preds = model_1(images)
                            _, class_predicted = torch.max(class_preds, 1)
                            total += class_targets.size(0)
                            correct += (class_predicted == class_targets).sum().item()
                            loss_val = class_criterion(class_preds, class_targets)
                            losses.append(loss_val.item())
                    validation_acc_1  = 100 * correct / total
                    validation_loss_1 = np.mean(losses) 
                    print(f'model_1_Epoch [{epoch + 1}/{num_epochs}] -   validation_acc_1: {validation_acc_1}-   validation_loss_1: {validation_loss_1}')
                    
                    history_1['train_acc_1'].append(train_acc_1)
                    history_1['train_loss_1'].append(train_loss_1)
                    history_1['validation_acc_1'].append(validation_acc_1)
                    history_1['validation_loss_1'].append(validation_loss_1)
                    
                    if validation_acc_1 > best_accuracy_1:
                        best_accuracy_1 = validation_acc_1
                        b_epoch_1    = epoch
    
                        best_model_state_dict_1 = copy.deepcopy(model_1.state_dict())
    
    
                    model_3.train()
                    losses = []
                    correct = 0
                    total = 0
    
                    for images, shape_targets, color_targets, symbol_targets, class_targets in train_dataloader:
                        # print(shape_targets)
                        optimizer_3.zero_grad()
                        shape_preds, color_preds, class_preds = model_3(images)

                        shape_loss = shape_criterion(shape_preds, shape_targets)
                        color_loss = color_criterion(color_preds, color_targets)
                        class_loss = class_criterion(class_preds, class_targets)  # Include class loss
                        # print('class_loss', class_loss)    
    
                        # Calculate the total loss by summing shape, color, symbol, and class losses
                        train_loss = shape_loss + color_loss + class_loss 
                        losses.append(train_loss.item())
                        total += class_targets.size(0)
                        # Backpropagation and optimization
                        train_loss.backward()
                        optimizer_3.step()
                        optimizer_3.zero_grad()
    
                        # Accumulate correct predictions
                        _, class_predicted = torch.max(class_preds, 1)
                        correct += torch.sum(class_predicted == class_targets).item()
                    
                    scheduler_3.step()
                    train_acc_3  = 100 * correct / total
                    train_loss_3 = np.mean(losses)  
                    print(f'model_3_Alpha [{alpha}] - Epoch [{epoch + 1}/{num_epochs}] -   train_acc_3: {train_acc_3} -   train_loss_3: {train_loss_3}')
            
                
                    # Validation loop for the model with all features
                    model_3.eval()
                    correct = 0
                    total = 0
                    losses = []
                    with torch.no_grad():
                        for images, shape_targets, color_targets, symbol_targets, class_targets in validation_dataloader:
                            shape_preds, color_preds, class_preds = model_3(images)
                            _, class_predicted = torch.max(class_preds, 1)
                            total += class_targets.size(0)
                            correct += (class_predicted == class_targets).sum().item()
                            loss_val = class_criterion(class_preds, class_targets)
                            losses.append(loss_val.item())
                    validation_acc_3  = 100 * correct / total
                    validation_loss_3 =  np.mean(losses) 
                    print(f'model_3_ Epoch [{epoch + 1}/{num_epochs}] -   validation_acc_3: {validation_acc_3}-   validation_loss_3: {validation_loss_3}')
    
    
                    history_3['train_acc_3'].append(train_acc_3)
                    history_3['train_loss_3'].append(train_loss_3)
                    history_3['validation_acc_3'].append(validation_acc_3)
                    history_3['validation_loss_3'].append(validation_loss_3)
    
    
                    if validation_acc_3 > best_accuracy_3:
                        best_accuracy_3 = validation_acc_3             
                        b_epoch_3    = epoch
    
                        best_model_state_dict_3 = copy.deepcopy(model_3.state_dict())
    
    
                print(f'b_epoch_1 [{b_epoch_1}]-best_accuracy_1: {best_accuracy_1} - b_epoch_3 [{b_epoch_3}]-  best_accuracy_3: {best_accuracy_3}')
    
                end1 = time.perf_counter()
                total_time = end1 - start1
                total_times.append(total_time) 
                print(f'Total Training Time for {alpha}: {total_time:.2f} seconds')
                
    
                
                # Save the best model's state dicts to separate files
                
                model_1_path     = os.path.join(save_dir, f'{subfolder}_best_model_1_itr_{itr}.pth')
                model_3_path     = os.path.join(save_dir, f'{subfolder}_alpha{alpha}_NoLogic_best_model_3_itr_{itr}.pth')
    
                torch.save(best_model_state_dict_1, model_1_path)
                torch.save(best_model_state_dict_3, model_3_path)
    
    
                history_folder = os.path.join(save_dir,'history')
                figure_folder  = os.path.join(save_dir,'figure')
    
                os.makedirs(history_folder, exist_ok=True)
                os.makedirs(figure_folder, exist_ok=True)
    
                history_file_path = os.path.join(history_folder, f'history_1_{subfolder}_alpha{alpha}_NoLogic_itr_{itr}.pkl')  
                with open(history_file_path, 'wb') as history_file:
                    pickle.dump(history_1, history_file)
                    
                history_file_path = os.path.join(history_folder, f'history_3_{subfolder}_alpha{alpha}_NoLogic_itr_{itr}.pkl')
                with open(history_file_path, 'wb') as history_file:
                    pickle.dump(history_3, history_file)
                
    
                plot_training_validation( history_1,history_3, filename=f'{subfolder_name}_{alpha}_{itr}.png')
    
                
            else:
                for Logic in Logics:
    
                    best_model_state_dict_3  = None
    
                    best_accuracy_3 = 0.0
    
                    history_3 = defaultdict(list)
    
                    shape_criterion  = nn.CrossEntropyLoss()
                    color_criterion  = nn.CrossEntropyLoss()
                    symbol_criterion = nn.CrossEntropyLoss()
                    class_criterion  = nn.CrossEntropyLoss() 
    
                    model_3  = CustomModel_3(num_shape_classes=shape_num, num_color_classes=colour_num, num_class_classes=class_num)  # Include num_class_classes
    
    
                    optimizer_3      = optim.Adam(model_3.parameters(), lr=0.001, weight_decay=1e-5)
                    scheduler_3     = lr_scheduler.StepLR(optimizer_3, step_size=7, gamma=0.1)
    
                    start2 = time.perf_counter()
                    for epoch in range(num_epochs):
                                # Training loop for the model with only class ID
                    
                        model_3.train()
                        losses = []
                        correct = 0
                        total = 0
    
                        for images, shape_targets, color_targets, symbol_targets, class_targets in train_dataloader:
                            # print(shape_targets)
                            optimizer_3.zero_grad()
                            shape_preds, color_preds, class_preds = model_3(images)
    

                            logic_loss = logical_loss(shape_preds, color_preds, class_preds, rules, Logic)
                            # print('logic_loss', logic_loss)
                            # Calculate the shape, color, symbol, and class losses
                            shape_loss = shape_criterion(shape_preds, shape_targets)
                            color_loss = color_criterion(color_preds, color_targets)
                            class_loss = class_criterion(class_preds, class_targets)  # Include class loss
                            # print('class_loss', class_loss)    
    
                            # Calculate the total loss by summing shape, color, symbol, and class losses
                            # train_loss = shape_loss + color_loss + class_loss + alpha * logic_loss
                            
                            # train_loss =class_loss+ (shape_loss + color_loss   )*logic_loss
                            train_loss =(shape_loss + color_loss + class_loss) + (logic_loss)


                            losses.append(train_loss.item())
                            total += class_targets.size(0)
                            # Backpropagation and optimization
                            train_loss.backward()
                            optimizer_3.step()
                            optimizer_3.zero_grad()
    
                            # Accumulate correct predictions
                            _, class_predicted = torch.max(class_preds, 1)
                            correct += torch.sum(class_predicted == class_targets).item()
                        
                        scheduler_3.step()
                        train_acc_3  = 100 * correct / total
                        train_loss_3 = np.mean(losses)  
                        print(f'model_3_Logic_{Logic} Alpha [{alpha}] - Epoch [{epoch + 1}/{num_epochs}] -   train_acc_3: {train_acc_3} -   train_loss_3: {train_loss_3}')
                
                    
                        # Validation loop for the model with all features
                        model_3.eval()
                        correct = 0
                        total = 0
                        losses = []
                        with torch.no_grad():
                            for images, shape_targets, color_targets, symbol_targets, class_targets in validation_dataloader:
                                shape_preds, color_preds, class_preds = model_3(images)
                                _, class_predicted = torch.max(class_preds, 1)
                                total += class_targets.size(0)
                                correct += (class_predicted == class_targets).sum().item()
                                loss_val = class_criterion(class_preds, class_targets)
                                losses.append(loss_val.item())
                        validation_acc_3  = 100 * correct / total
                        validation_loss_3 =  np.mean(losses) 
                        print(f'Epoch [{epoch + 1}/{num_epochs}] -   validation_acc_3: {validation_acc_3}-   validation_loss_3: {validation_loss_3}')
    
    
                        history_3['train_acc_3'].append(train_acc_3)
                        history_3['train_loss_3'].append(train_loss_3)
                        history_3['validation_acc_3'].append(validation_acc_3)
                        history_3['validation_loss_3'].append(validation_loss_3)
    
    
                        if validation_acc_3 > best_accuracy_3:
                            best_accuracy_3 = validation_acc_3             
                            b_epoch_3    = epoch
    
                            best_model_state_dict_3 = copy.deepcopy(model_3.state_dict())
    
    
                    print(f'b_epoch_1 [{b_epoch_1}]-best_accuracy_1: {best_accuracy_1} - b_epoch_3 [{b_epoch_3}]-  best_accuracy_3: {best_accuracy_3}')
    
                    end2 = time.perf_counter()
                    total_time = end2 - start2
                    total_times.append(total_time) 
                    print(f'Total Training Time for {Logic}: {total_time:.2f} seconds')
                    

                    model_3_path     = os.path.join(save_dir, f'{subfolder}_alpha{alpha}_Logic_{Logic}_best_model_3_itr_{itr}.pth')
                    torch.save(best_model_state_dict_3, model_3_path)
    
                    history_folder = os.path.join(save_dir,'history')
                    figure_folder  = os.path.join(save_dir,'figure')
    
                    os.makedirs(history_folder, exist_ok=True)
                    os.makedirs(figure_folder, exist_ok=True)
    

                        
                    history_file_path = os.path.join(history_folder, f'history_3_{subfolder}_alpha{alpha}_Logic{Logic}_itr_{itr}.pkl')
                    with open(history_file_path, 'wb') as history_file:
                        pickle.dump(history_3, history_file)
                    
    
                    plot_training_validation( history_1,history_3, filename=f'{subfolder_name}_{Logic}_{alpha}_{itr}.png')
    
    plt.figure(figsize=(10, 6))
    plt.bar(subfolders, total_times, color='blue')
    plt.xlabel('Subfolder')
    plt.ylabel('Total Training Time (seconds)')
    plt.title('Total Training Time for Each Subfolder')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    filename = 'total_times_plot.png'
    plot_save_path = os.path.join(figure_folder, filename)
    plt.savefig(plot_save_path)
    plt.show()
    
