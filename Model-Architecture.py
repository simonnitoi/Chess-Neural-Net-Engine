import torch
import torch.nn
import torch.nn.functional

class SEResBlock(torch.nn.Module):

    def __init__(self):
        super(SEResBlock, self).__init__()
        self.channels = 128
        self.se_channels = 32

        self.cv1 = torch.nn.Conv2d(self.channels,self.channels,3,padding=1,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.channels)
        self.cv2 = torch.nn.Conv2d(self.channels,self.channels,3,padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(self.channels)

        # SE layers
        self.se_fc1 = torch.nn.Linear(self.channels, self.se_channels)
        self.se_fc2 = torch.nn.Linear(self.se_channels, self.channels * 2)

    def forward(self,x):
        identity = x
        x = torch.nn.functional.relu(self.bn1(self.cv1(x)))
        x = self.bn2(self.cv2(x))


        # SE layers
        se = x.mean(dim=(2,3))

        se = torch.nn.functional.relu(self.se_fc1(se))
        se = self.se_fc2(se)

        se = se.unsqueeze(-1).unsqueeze(-1)

        gamma, beta = se.chunk(2, dim=1)

        x = torch.sigmoid(gamma) * x + beta


        x = x + identity
        x = torch.nn.functional.relu(x)
        return x

class ChessNet(torch.nn.Module):

    def __init__(self):
        super(ChessNet, self).__init__()
        self.blocks = 10

        # SHARED
        self.cv1 = torch.nn.Conv2d(17,128,3,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(128)

        self.layers = torch.nn.ModuleList()
        for i in range(self.blocks):
            self.layers.append(SEResBlock())

        # VALUE HEAD
        self.Vcv1 = torch.nn.Conv2d(128,16,3,padding=1)
        self.Vbn1 = torch.nn.BatchNorm2d(16)

        self.Vln1 = torch.nn.Linear(1024,64)
        self.Vln2 = torch.nn.Linear(64,1)

        # POLICY HEAD
        self.Pcv1 = torch.nn.Conv2d(128,32,3,padding=1)
        self.Pbn1 = torch.nn.BatchNorm2d(32)
        self.Pln1 = torch.nn.Linear(2048,4609)
        
    def forward(self,x):

        # SHARED
        x = torch.nn.functional.relu(self.bn1(self.cv1(x)))

        for layer in self.layers:
            x = layer(x)

        # VALUE HEAD
        val = torch.nn.functional.relu(self.Vbn1(self.Vcv1(x)))
        val = torch.flatten(val, start_dim=1)
        val = torch.nn.functional.relu(self.Vln1(val))
        val = torch.tanh(self.Vln2(val))

        # POLICY HEAD
        pol = torch.nn.functional.relu(self.Pbn1(self.Pcv1(x)))
        pol = torch.flatten(pol, start_dim=1)
        pol = self.Pln1(pol)

        return val,pol