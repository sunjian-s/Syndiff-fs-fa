import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 核心 UNetplus 定义 (从你提供的代码提取) ---
class UNetplus(nn.Module):
    def __init__(self, inDim=3): # 默认改成3，防止你只有RGB
        super(UNetplus,self).__init__()
        self.inDim = inDim
        self.ReLU = nn.ReLU()
        # ConLevel(Fsize,inC,outC)
        self.ConLevel00 = ConLevel(3, inDim, 32)
        self.ConLevel001 = ConLevel(3,32,32)
        self.ConLevel10 = ConLevel(3,32,64)
        self.ConLevel20 = ConLevel(3,64,128)
        self.ConLevel201 = ConLevel(3,128,128)
        self.ConLevel30 = ConLevel(3,128,256)
        self.ConLevel301 = ConLevel(3,256,256)
        self.ConLevel40 = ConLevel(3,256,512)
        self.ConLevel401 = ConLevel(3,512,512)
        self.upsample10 = Upsample(2,64)
        self.ConLevel01 = ConLevel(3,32*2,32)
        self.ConLevelout1 = ConLevel(3,32,5)
        self.upsample20 = Upsample(2,128)
        self.ConLevel11 = ConLevel(3,64*2,64)
        self.upsample11 = Upsample(2,64)
        self.ConLevel02 = ConLevel(3,32*3,32)
        self.ConLevelout2 = ConLevel(3,32,5)
        self.upsample30 = Upsample(2,256)
        self.ConLevel21 = ConLevel(3,128*2,128)
        self.upsample21 = Upsample(2,128)
        self.ConLevel12 = ConLevel(3,64*3,64)
        self.upsample12 = Upsample(2,64)
        self.ConLevel03 = ConLevel(3,32*4,32)
        self.ConLevelout3 = ConLevel(3,32,5)
        self.upsample40 = Upsample(2,512)
        self.ConLevel31 = ConLevel(3,256*2,256)
        self.upsample31 = Upsample(2,256)
        self.ConLevel22 = ConLevel(3,128*3,128)
        self.upsample22 = Upsample(2,128)
        self.ConLevel13 = ConLevel(3,64*4,64)
        self.upsample13 = Upsample(2,64)
        self.ConLevel04 = ConLevel(3,32*5,32)
        self.ConLevelout4 = ConLevel(3,32,5)
        self.output4 = nn.Softmax2d() 

    def forward(self,X):
        output = []
        x_00 = self.ConLevel00(X)
        x_00 = self.ConLevel001(x_00)
        x_10 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_00)
        x_10 = self.ConLevel10(x_10)
        x_20 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_10)
        x_20 = self.ConLevel20(x_20)
        x_20 = self.ConLevel201(x_20)

        x_30 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_20)
        x_30 = self.ConLevel30(x_30)
        x_30 = self.ConLevel301(x_30)

        x_40 = nn.MaxPool2d(2, 2, padding=(2 - 1) // 2)(x_30)
        x_40 = self.ConLevel40(x_40)
        x_40 = self.ConLevel401(x_40)

        up10 = self.upsample10(x_10)
        x_01 = self.ConLevel01(torch.cat([x_00,up10],1))
        output1 = self.ConLevelout1(x_01)
        output1 = F.softmax(output1, dim=1)

        up20 = self.upsample20(x_20)
        x_11 = self.ConLevel11(torch.cat([x_10,up20],1))
        up11 = self.upsample11(x_11)
        x_02 = self.ConLevel02(torch.cat([x_00,x_01,up11],1))
        output2 = self.ConLevelout2(x_02)
        output2 = F.softmax(output2, dim=1)

        up30 = self.upsample30(x_30)
        x_21 = self.ConLevel21(torch.cat([x_20,up30],1))
        up21 = self.upsample21(x_21)
        x_12 = self.ConLevel12(torch.cat([x_10,x_11,up21],1))
        up12 = self.upsample12(x_12)
        x_03 = self.ConLevel03(torch.cat([x_00,x_01,x_02,up12],1))
        output3 = self.ConLevelout3(x_03)
        output3 = F.softmax(output3, dim=1)

        up40 = self.upsample40(x_40)
        x_31 = self.ConLevel31(torch.cat([x_30,up40],1))
        up31 = self.upsample31(x_31)
        x_22 = self.ConLevel22(torch.cat([x_20,x_21,up31],1))
        up22 = self.upsample22(x_22)
        x_13 = self.ConLevel13(torch.cat((x_10,x_11,x_12,up22),1))
        up13 = self.upsample13(x_13)
        x_04 = self.ConLevel04(torch.cat([x_00,x_01,x_02,x_03,up13],1))
        output4 = self.ConLevelout4(x_04)
        output4 = self.output4(output4)

        output.append(output1)
        output.append(output2)
        output.append(output3)
        output.append(output4)
        # 只返回最后一个 output4 (Shape: [B, 5, H, W])
        return output[-1] 

class ConLevel(nn.Module):
    def __init__(self,Fsize,inC,outC):
        super(ConLevel,self).__init__()
        self.Fsize = Fsize
        self.inC = inC
        self.outC = outC
        self.conv = nn.Sequential(nn.Conv2d(self.inC, self.outC, kernel_size=self.Fsize, stride=1, padding=self.Fsize//2, bias=True),
                                  nn.BatchNorm2d(outC, affine=True),
                                  nn.ReLU(inplace=True)
                                 )
    def forward(self,X):
        return self.conv(X)

class Upsample(nn.Module):
    def __init__(self, scale, indim):
        super(Upsample, self).__init__()
        self.scale = scale
        self.indim = indim
        self.up = nn.Sequential(torch.nn.ConvTranspose2d(indim,indim//2,kernel_size=scale*2,stride=scale,padding=scale//2),
                                nn.ReLU())
    def forward(self, x):
        return self.up(x)