import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=9):
        super(Net, self).__init__()

        def ConvBlock(c_in, c_out, size, pd=1):
            conv = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=size, stride=1, padding=pd),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
            )
            return conv 

        self.conv1 = ConvBlock(1, 16, 3, 1)
        self.conv2 = ConvBlock(16, 64, 3, 1)    # Res
        self.conv3 = ConvBlock(64, 64, 3, 1)
        self.conv4 = ConvBlock(64, 64, 3, 1)    # Res
        self.conv5 = ConvBlock(64, 16, 3, 1)
        self.conv6 = ConvBlock(16, 4, 3, 1)
        self.dropout = nn.Dropout2d(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.num_classes = num_classes
        # 3 input image channel, 16 output channels, 3x3 square convolution kernel
        
        self.fc1 = nn.Linear(4*16*16,9)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()



    def forward(self, x):
        x_id = x  # 128
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x_id+x)
        x = self.dropout(x)
        x = self.pool(x)    # 64ch
        x_id = x 
        x = self.conv3(x)
        x = self.conv4(x) 
        x = F.relu(x_id+x)
        x = self.dropout(x)
        x = self.pool(x)    # 32
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dropout(x)
        x = self.pool(x)    # 16
        # print(x.shape)
        x = x.view(-1,4*16*16) # Flatten layer
        # x = F.silu(self.fc1(x))
        x = self.fc1(x)
        return x
