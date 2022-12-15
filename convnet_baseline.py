import os
from os.path import join
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import gc
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34, resnet50
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")
# Control Randomness
import random
random_seed = 7
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import *
from dataset import WaferDataset
from net import Net
from torch.utils.tensorboard import SummaryWriter
import datetime 
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"]="1"
# Config


RE_SIZE = 128
INPUT_CH = 1    # Must be 1
NUM_CLASSES = 9
IS_PRETRAINED = False
LEARNING_RATE = 1e-3
EPOCHS = 100

parser = argparse.ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=256)
parser.add_argument('-lr', '--lr',type=float, default=1e-3)

parser.add_argument('-augprob', '--prob', type=float, default=0.0, help='Probability of conduct transforms')
parser.add_argument('-use-downsample','--downsample', type=bool, default=False, help='Whether to use downsampled dataset')
parser.add_argument('-use-cweight','--cweight', type=bool, default=False)
parser.add_argument('-use-focal','--focal', type=bool, default=False)
parser.add_argument('-ckpt', '--ckpt', type=str)


args = parser.parse_args()

ROOT = '/mnt/c/Users/cjs98/Drive/Codes/TNT/22-1-Industrial-AI/TASK2/'

# Load data
if args.downsample==True:
    print("Using Undersampling")
    X_train = torch.load(ROOT+'Data/X_train_u.pt')
    y_train = torch.load(ROOT+'Data/y_train_u.pt')
else:
    X_train = torch.load(ROOT+'Data/X_train.pt')
    y_train = torch.load(ROOT+'Data/y_train.pt')

X_val = torch.load(ROOT+'Data/X_val.pt')
X_test = torch.load(ROOT+'Data/X_test.pt')
y_val = torch.load(ROOT+'Data/y_val.pt')
y_test = torch.load(ROOT+'Data/y_test.pt')


device = torch.device("cuda")

# Transforms and Dataset

train_transform, else_transform = get_transforms(size=RE_SIZE, p=args.prob) # Just Resize and tensorize

train_dataset = WaferDataset(X_train, y_train, transform=train_transform)
val_dataset = WaferDataset(X_val, y_val, transform=else_transform)
test_dataset = WaferDataset(X_test, y_test, transform=else_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

## MODEL

model = Net(num_classes=NUM_CLASSES)

# model = resnet34(pretrained=False)
# model.conv1 = nn.Conv2d(INPUT_CH, 64, kernel_size=(1, 1), stride=(2, 2), padding=(3, 3), bias=False)
# feat = model.fc.in_features
# model.fc = nn.Linear(feat, NUM_CLASSES)




if args.cweight==True:
    print("Using Class Weight")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights = torch.tensor(class_weights,dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if args.focal==True:
        print("Using Focal Loss")
        criterion = FocalLoss(weights=weights.to(device), gamma=2, reduce=True)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=weights.to(device))
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.focal==True:
        print("Using Focal Loss")
        criterion = FocalLoss(weights=None, gamma=2, reduce=True)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

now = datetime.datetime.now().strftime("%m%d-%H%M")
save_name = f'convnet_b{args.batch_size}_100ep'
save_path = f'models/{save_name}_{now}.pt'

log_name = "logs/" + save_name + "/" + now
writer = SummaryWriter(log_dir=log_name)

print(save_name)
print(save_path)

model.to(device)

if __name__ == '__main__':

    print(args)
    # if args.ckpt:
    #     pass
    # else:
    # print('Training')
    # for epoch in range(EPOCHS):
    #     train(model, train_loader, criterion, optimizer, epoch, EPOCHS, device, writer)
    #     evaluate(model, val_loader, criterion, epoch, device, writer, save_path)


    # print(f'{max(val_acc_list):.5f}, {max(val_f1_macro_list):.5f}')

    # max_val_acc = np.round(max(val_acc_list), 4)
    # max_val_f1_macro = np.round(max(val_f1_macro_list), 4)
    # max_val_f1_weighted = np.round(max(val_f1_weighted_list), 4)

    # Test and Write
    # model = resnet34(pretrained=False)
    # model.conv1 = nn.Conv2d(INPUT_CH, 64, kernel_size=(1, 1), stride=(2, 2), padding=(3, 3), bias=False)
    # feat = model.fc.in_features
    # model.fc = nn.Linear(feat, NUM_CLASSES)

    model = Net(num_classes=NUM_CLASSES)

    # LOAD trained model 
    model.load_state_dict(torch.load(args.ckpt))
    print('ckpt Load Success')
    model.to(device)
    test_acc, test_f1_macro, test_f1_weighted = test(model, test_loader, device)

    # result = pd.read_csv('result.csv', header=0)

    # result.iloc[0, 6]= max_val_acc
    # result.iloc[0, 7] = max_val_f1_macro
    # result.iloc[0, 8] = max_val_f1_weighted
    # result.iloc[0, 9] = test_acc
    # result.iloc[0, 10] = test_f1_macro
    # result.iloc[0, 11] = test_f1_weighted
    # result.to_csv('result.csv', index=False)
