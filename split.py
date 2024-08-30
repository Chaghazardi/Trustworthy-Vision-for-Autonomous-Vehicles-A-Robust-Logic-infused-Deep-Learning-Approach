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





# Define the root folder containing subfolders (each with its own CSV file)
root_folder = 'datasets/train2'
test_folder = 'datasets/test2'

    
# List subfolders (categories) in the root folder
subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
print(subfolders)

# Loop through each subfolder (category)
for subfolder in subfolders:

    subfolder_path = os.path.join(root_folder, subfolder)
    print(subfolder_path)
    data_path = os.path.join(subfolder_path, f'{subfolder}.csv')
    data = pd.read_csv(data_path)
    # print(len(data))
    classID = data.columns[1]
    unique_class_ids = data[classID].unique()

    train_data = pd.DataFrame(columns=data.columns)
    validation_data = pd.DataFrame(columns=data.columns)

    # Split the data for each class ID and each dataset size
    for class_id in unique_class_ids:
        class_data = data[data[classID] == class_id]
        # print(class_data)
        train_data_temp, validation_data_temp = train_test_split(class_data, test_size=0.2, random_state=42)

        train_data      = pd.concat([train_data, train_data_temp])
        validation_data = pd.concat([validation_data, validation_data_temp])

    # Save the split datasets into separate CSV files within the dataset folder
    train_data.to_csv(os.path.join(subfolder_path, 'train_dataset.csv'), index=False)
    validation_data.to_csv(os.path.join(subfolder_path, 'validation_dataset.csv'), index=False)

