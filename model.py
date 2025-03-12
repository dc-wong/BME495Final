import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class Separable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding)
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        #x = checkpoint(self.depthwise, x, use_reentrant=False)
        #x = checkpoint(self.pointwise, x, use_reentrant=False)
        x = self.pointwise(self.depthwise(x))
        return x

class ConvSet(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.conv1 = Separable(in_channels = in_channels, out_channels = middle_channels)
        self.conv2 = Separable(in_channels = middle_channels, out_channels = out_channels)
        self.dropout = nn.Dropout3d(p=dropout_prob)
        self.relu = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm3d(out_channels)
    
    def forward(self, x):
        x = self.dropout(self.conv2(self.conv1(x)))
        x = self.relu(x)
        x = self.batchnorm(x)
        return x
        

class MiniModel(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.conv1 = ConvSet(1, 32, 64)
        self.conv2 = ConvSet(64, 64, 128)
        self.conv3 = ConvSet(128, 128, 256)

        self.maxPool = nn.AvgPool3d(kernel_size = 2, stride = 2)

        self.deconv1 = ConvSet(128 + 256, 128, 128)
        self.deconv2 = ConvSet(64 + 128, 64, 64)
        
        self.last = nn.Conv3d(in_channels = 64, out_channels = 1, kernel_size = 1)

        self.dropout = nn.Dropout3d(p=dropout_prob)
    
    def forward(self, x):
        x1 = self.conv1(x)
        # print(x1.size())
        x2 = checkpoint(self.maxPool, x1)
        x2 = self.conv2(x2)
        
        # print(x2.size())
        y = checkpoint(self.maxPool,x2)
        y = self.conv3(y)
        
        # print(x3.size())
        y = self.dropout(y)
        y = F.interpolate(y, size=x2.shape[2:], mode='trilinear', align_corners=True)

        x2 = self.dropout(x2)
        y = self.deconv1(torch.cat((x2, y), dim = 1))
        y = self.dropout(y)
        y = F.interpolate(y, size=x1.shape[2:], mode='trilinear', align_corners=True)

        x1 = self.dropout(x1)
        y = self.deconv2(torch.cat((x1, y), dim = 1))
        y = self.last(y)
        return y
    
class MainModel(nn.Module):
    def __init__(self, weights = r"C:\PROJECTS\BME495\FinalProject\weights\best_mini_model_params.pt", dropout_prob = 0.5):
        super().__init__()
        self.pretrained = MiniModel()
        self.pretrained.load_state_dict(torch.load(weights, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        for param in self.pretrained.parameters():
            param.requires_grad = False


        self.conv1 = ConvSet(2, 32, 64)
        # self.conv1 = nn.Conv3d(in_channels = 2, out_channels = 1, kernel_size = 5, padding=2)
        # self.conv2 = ConvSet(64, 64, 128)

        # self.maxPool = nn.MaxPool3d(kernel_size = 2, stride = 2)

        # self.deconv1 = ConvSet(64 + 128, 64, 64)

        # self.gate1 = LinearAttention(128, 32, 64)

        self.output = nn.Conv3d(in_channels = 64, out_channels = 1, kernel_size = 1)
        self.dropout = nn.Dropout3d(p=dropout_prob)
        # self.output = nn.Sigmoid()
    
    def forward(self, x):
        # print(x.shape)
        y = F.avg_pool3d(x, kernel_size=4, ceil_mode=True)
        # print(y.shape)
        y = self.pretrained(y)
        # print(y.shape)
        y = self.dropout(y)
        y = F.interpolate(y, size=x.shape[2:], mode='trilinear', align_corners=True)
        # print(x.shape, y.shape)
        y = self.conv1(torch.cat((x, y), dim = 1))
        # y = self.conv2(self.maxPool(x1))
        # y = F.interpolate(y, size=x1.shape[2:], mode='trilinear', align_corners=True)
        # a1 = self.gate1(x1, y)
        # y = self.deconv1(torch.cat((a1, y), dim=1))

        # y = self.last(y)
        y = self.output(y)
        return y
    

class SingleModel(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        z = 32
        self.scale = 2
        self.conv1 = ConvSet(1, z, 2 * z)
        self.conv2 = ConvSet(2 * z, 2 * z, 4 * z)
        self.conv3 = ConvSet(4 * z, 4 * z, 8 * z)
        self.conv4 = ConvSet(8 * z, 8 * z, 16 * z)

        self.maxPool = nn.MaxPool3d(kernel_size = self.scale, stride = self.scale)

        self.deconv1 = ConvSet(24 * z, 8 * z, 8 * z)
        self.deconv2 = ConvSet(12 * z, 4 * z, 4 * z)
        self.deconv3 = ConvSet(6 * z, 2 * z, 2 * z)
        
        self.last = Separable(in_channels = 2 * z, out_channels = 1, kernel_size = 1, padding=0) #should be 2*z, but we are just testing to see if it can learn
        self.output = nn.Sigmoid()
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxPool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxPool(x2)
        x3 = self.conv3(x3)
        y = self.maxPool(x3)
        y = self.conv4(y)

        y = F.interpolate(y, scale_factor = self.scale)
        y = self.deconv1(torch.cat((x3, y), dim = 1))

        y = F.interpolate(y, scale_factor = self.scale)
        y = self.deconv2(torch.cat((x2, y), dim = 1))

        y = F.interpolate(y, scale_factor = self.scale)
        y = self.deconv3(torch.cat((x1, y), dim = 1))

        y = self.last(y)
        y = self.output(y)
        return y

class MultiModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        z = 32
        self.pretrained = SingleModel()
        self.pretrained.load_state_dict(torch.load(weights, weights_only=True))
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.pretrained.last = Separable(in_channels = 2 * z, out_channels = z *2, kernel_size = 1, padding=0)
    
    def forward(self, x):
        return self.pretrained(x)
