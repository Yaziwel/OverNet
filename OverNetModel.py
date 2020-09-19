import torch
from torch import nn, optim
import random
import torch.nn.functional as F

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class AdaptiveAdd(nn.Module):
    def __init__(self):
        super(AdaptiveAdd,self).__init__()
        self.lambda_1 = nn.Parameter(torch.ones(1))
        self.lambda_2 = nn.Parameter(torch.ones(1))
    def forward(self, se_out, skip_layer ):
        # print(self.lambda_1.requires_grad)
        return se_out*self.lambda_1+skip_layer*self.lambda_2


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self,channels):
        super(ResBlock,self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        expand = 6
        linear = 0.8
        ##WideActivation
        self.WA=nn.Sequential(
            wn(nn.Conv2d(channels, channels*expand, 1)),
            nn.ReLU(True),
            wn(nn.Conv2d(channels*expand, int(channels*linear), 1)),
            wn(nn.Conv2d(int(channels*linear), channels, 3,padding=1))
            )
        ##Squeeze and Excitation
        self.se = SE_Block(channels)
        self.add=AdaptiveAdd()
    def forward(self,f_map):
        out = self.WA(f_map)
        out = self.se(out)
        out = self.add(out, f_map)
        return out

class DenseGroup(nn.Module):
    def __init__(self,channels):
        super(DenseGroup,self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.resblock1 = ResBlock(channels)
        self.conv1 = wn(nn.Conv2d(2*channels,channels,1))
        self.resblock2 = ResBlock(channels)
        self.conv2 = wn(nn.Conv2d(3*channels,channels,1))
        self.resblock3 = ResBlock(channels)
    def forward(self, f_map):
        concat=f_map
        rb=self.resblock1(concat)
        concat=torch.cat([concat, rb],dim=1)
        rb=self.resblock2(self.conv1(concat))
        concat=torch.cat([concat, rb],dim=1)
        rb=self.resblock3(self.conv2(concat))
        return rb

class OverNet(nn.Module):
    def __init__(self,channels=16):
        super(OverNet,self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)

        #NEW
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)


        self.conv3x3=wn(nn.Conv2d(3,channels,3,padding=1))
        self.DG1=DenseGroup(channels)
        self.conv1x1_1=wn(nn.Conv2d(2*channels,channels,1))
        self.DG2=DenseGroup(channels)
        self.conv1x1_2=wn(nn.Conv2d(3*channels,channels,1))
        self.DG3=DenseGroup(channels)
        self.conv1x1_3=wn(nn.Conv2d(3*channels,channels,1))
        self.longRangeSkip=nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        wn(nn.Conv2d(channels,channels,1)),
        nn.ReLU(True)
        )
        self.add=AdaptiveAdd()

        self.OSMBlock=nn.Sequential(
            wn(nn.Conv2d(channels, 1600,3,padding=1)),
            nn.PixelShuffle(5),
            wn(nn.Conv2d(64,3,3,padding=1))
            )


    def forward(self, img):

        img = self.sub_mean(img)

        f1=self.conv3x3(img)
        f2=self.DG1(f1)
        f3=self.DG2(self.conv1x1_1(torch.cat([f1,f2],dim=1)))
        f4=self.DG3(self.conv1x1_2(torch.cat([f1,f2,f3],dim=1)))

        extract_f1=self.longRangeSkip(f1)
        extract_f2=self.conv1x1_3(torch.cat([f2,f3,f4],dim=1))

        h=self.add(extract_f1, extract_f2)
        h=self.OSMBlock(h)
        f_x4_1=F.interpolate(h, size=[img.shape[2]*4, img.shape[3]*4], mode="bicubic",align_corners=False)
        f_x4_2=F.interpolate(img, size=[img.shape[2]*4, img.shape[3]*4], mode="bicubic",align_corners=False)
        out=f_x4_1+f_x4_2

        out = self.add_mean(out)
        return out


if __name__ == "__main__":
    import pickle
    with open("E:\\STUDY\\Places\\EPS\OverNet\\test\\0802_1.bin","rb") as f:
        img=pickle.load(f)
    img=img.unsqueeze(0).type(torch.FloatTensor)
    print(img.shape)
    G=OverNet()
    pred=G(img)
