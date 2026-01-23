from torch import nn
import torch.nn.functional as F
import torch
import math

def get_2d_positional_encoding(height, width, d_model):
    """
    Generates a 2D sinusoidal positional encoding tensor.
    
    Args:
        height (int): The height of the 2D grid.
        width (int): The width of the 2D grid.
        d_model (int): The dimension of the encoding vectors (must be divisible by 4).
        
    Returns:
        torch.Tensor: A tensor of shape (height, width, d_model) with positional encodings.
    """
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for 2D encoding.")
        
    # Create position indices
    pos_x = torch.arange(height, dtype=torch.float32).unsqueeze(1).repeat(1, width)
    pos_y = torch.arange(width, dtype=torch.float32).unsqueeze(0).repeat(height, 1)

    # Create dimension indices for the frequencies
    # d_model is split into 4 parts: sin(x), cos(x), sin(y), cos(y)
    d_model_half = d_model // 2
    d_model_quarter = d_model // 4
    
    # Calculate the division term for the frequencies
    div_term = torch.exp(torch.arange(0., d_model_quarter, 1.) * -(math.log(10000.0) / d_model_quarter))
    
    # Calculate 1D encodings for x and y
    pe_x = torch.zeros(height, width, d_model_half)
    pe_y = torch.zeros(height, width, d_model_half)
    
    # Apply sine and cosine to even and odd indices for x dimension
    pe_x[:, :, 0::2] = torch.sin(pos_x.unsqueeze(-1) * div_term)
    pe_x[:, :, 1::2] = torch.cos(pos_x.unsqueeze(-1) * div_term)
    # Apply sine and cosine to even and odd indices for y dimension
    pe_y[:, :, 0::2] = torch.sin(pos_y.unsqueeze(-1) * div_term)
    pe_y[:, :, 1::2] = torch.cos(pos_y.unsqueeze(-1) * div_term)
    # Concatenate the x and y encodings along the last dimension
    pe = torch.cat([pe_x, pe_y], dim=-1)
    return pe

class LRGenerator(nn.Module):
    def __init__(self, PatchSize, Headhdim,N,nchann):
        super().__init__()
        self.rank = 3
        self.npatch = N//PatchSize
        self.psize = PatchSize
        self.vtokenproj = nn.Linear(PatchSize**2,Headhdim)
        self.htokenproj = nn.Linear(PatchSize**2,Headhdim)
        self.Vprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N) for i in range(self.rank)])
        self.Hprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N) for i in range(self.rank)])
        pe = get_2d_positional_encoding(N,N,nchann)
        pe = get_2d_positional_encoding(N, N, nchann)      # (N, N, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)              # (1, C, N, N)
        self.register_buffer("pos_enc", pe)
    def forward(self, x):
        ## Patching and tokenization
        b,c,h,w = x.shape
        print(x.shape)
        print(self.pos_enc.shape)
        y = x + self.pos_enc # B x C x N x N
        p = self.psize
        np = self.npatch
        patches = y.unfold(2, p, p).unfold(3, p, p) ## B x Np x Np x Ps x Ps
        #patches = patches.contiguous().view(3, -1, p, p)
        patches = patches.contiguous().view(b,c,np,np,p*p) ## flattened patches
        lrfeats = torch.zeros_like(x)
        for _c in range(c):
            VCat = []
            #vertical component
            for i in range(np):
                vproj = 0.0
                for j in range(np):
                    vproj += self.vtokenproj(patches[:,_c,j,i])
                VCat.append(vproj)
            vcomp = torch.concat(VCat,dim=-1)
            
            HCat = []
            #horizontal component
            for j in range(np):
                hproj = 0.0
                for i in range(np):
                    hproj += self.htokenproj(patches[:,_c,j,i])
                HCat.append(hproj)
            hcomp = torch.concat(HCat,dim=-1)

            Vs = []
            Hs = []
            for r in range(self.rank):
                Vs.append(self.Vprojs[r](vcomp))
                Hs.append(self.Hprojs[r](hcomp))
            
            V = torch.stack(Vs,dim=-1)
            H = torch.stack(Hs,dim=-1)
            #feat = torch.matmul(V,H)
            feat = V@H.transpose(-1,-2)
            lrfeats[:,_c,:,:] = feat
        return lrfeats
        

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, in_planes//2, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(in_planes//2)
        self.conv2 = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(in_planes//2)
        self.conv3 = nn.Conv2d(in_planes//2, out_planes, kernel_size=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            # In pre-act ResNet, shortcut conv happens on the *activated* input
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))

        shortcut = self.shortcut(out) if self.shortcut is not None else x

        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))

        out = out + shortcut
        return out


class PreActBottleneckLR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1,use_lr=False,N=32):
        super().__init__()
        
        self.lrgen = LRGenerator(4,4,N,in_planes) if use_lr else None
        self.convlr = nn.Conv2d(in_planes, out_planes//2, kernel_size=1, bias=True)

        self.convlrmatch = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=1, bias=True)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, in_planes//2, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(in_planes//2)
        self.conv2 = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(in_planes//2)
        self.conv3 = nn.Conv2d(in_planes//2, out_planes//2, kernel_size=1, bias=True)
        self.stride = stride
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            # In pre-act ResNet, shortcut conv happens on the *activated* input
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        if self.lrgen is not None:
            lrfeats = self.lrgen(x)
            lrfeats = self.convlr(lrfeats)
            if self.stride == 2:
                lrfeats = F.avg_pool2d(lrfeats, kernel_size=2, stride=2)
        else:
            lrfeats = self.convlrmatch(out) # self.convlr(out)

        out = torch.concat([out,lrfeats],dim=1)

        out = out + shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,128,  num_blocks[0], stride=1,N=128) #512
        self.layer2 = self._make_layer(block, 128,256, num_blocks[1], stride=2,N=128) #128
        self.layer3 = self._make_layer(block, 256,512, num_blocks[2], stride=2,N=64) # 64
        self.layer4 = self._make_layer(block, 512,512*4, num_blocks[3], stride=2,N=32) #32

        self.bn_final = nn.BatchNorm2d(512 * 4)
        self.fc = nn.Linear(512 * 4, num_classes)

    """def _make_layer(self, block, iplanes, oplanes, nblocks, stride,N):
        strides = [stride] + [1] * (nblocks - 1)
        layers = []
        in_ch = iplanes
        for i,s in enumerate(strides):
            layers.append(block(in_ch, oplanes, stride=s,use_lr=True if i==0 else False,N=N))
            in_ch = oplanes
        return nn.Sequential(*layers)
    """
    def _make_layer(self, block, iplanes, oplanes, nblocks, stride, N):
        strides = [stride] + [1] * (nblocks - 1)
        layers = []
        in_ch = iplanes
        curN = N

        for i, s in enumerate(strides):
            layers.append(block(
                in_ch, oplanes,
                stride=s,
                use_lr=(i == 0),
                N=curN
            ))
            in_ch = oplanes
            if s == 2:
                curN = curN // 2  # spatial downsample happens here

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out


def preact_resnet18_bottleneck(num_classes=1000, in_ch=3):
    # "ResNet-18 depth schedule" but bottleneck blocks
    return PreActResNet(PreActBottleneckLR, [2, 2, 2, 2], num_classes=num_classes, in_ch=in_ch)