from torch import nn
import torch.nn.functional as F
import torch
import math
import copy
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


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

def get_1d_positional_encoding(length, d_model):
    """
    Generates a 2D sinusoidal positional encoding tensor.
    
    Args:
        height (int): The height of the 2D grid.
        width (int): The width of the 2D grid.
        d_model (int): The dimension of the encoding vectors (must be divisible by 4).
        
    Returns:
        torch.Tensor: A tensor of shape (height, width, d_model) with positional encodings.
    """
    #if d_model % 4 != 0:
    #    raise ValueError("d_model must be divisible by 4 for 2D encoding.")
        
    # Create position indices
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    # Create dimension indices for the frequencies
    # d_model is split into 4 parts: sin(x), cos(x), sin(y), cos(y)
    d_model_half = d_model // 2
    
    # Calculate the division term for the frequencies
    div_term = torch.exp(torch.arange(0., d_model, 2.) * -(math.log(10000.0) / d_model))
    
    # Calculate 1D encodings 
    pe = torch.zeros(length, d_model)
    
    # Apply sine and cosine to even and odd indices for x dimension
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


class LRGenerator(nn.Module):
    def __init__(self, PatchSize, Headhdim,N,nchann):
        super().__init__()
        self.rank = 3
        self.npatch = N//PatchSize
        self.psize = PatchSize
        self.vtokenproj = nn.Linear(PatchSize**2,Headhdim)
        self.htokenproj = nn.Linear(PatchSize**2,Headhdim)
        self.Vprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N*rank) for i in range(self.rank)])
        self.Hprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N*rank) for i in range(self.rank)])
        pe = get_2d_positional_encoding(N,N,nchann)
        pe = get_2d_positional_encoding(N, N, nchann)      # (N, N, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)              # (1, C, N, N)
        self.register_buffer("pos_enc", pe)

class _lindpout3d(nn.Module):
    def __init__(self,p:float = 0.2):
        super().__init__()
        self.p = p
        self.drop = nn.Dropout3d(self.p)
    def forward(self,x):
        y = torch.permute(x,(0,1,4,2,3))
        y = self.drop(y)
        y = torch.permute(y,(0,1,3,4,2))
        return y
    
class _lindpout1d(nn.Module):
    def __init__(self,p:float = 0.2):
        super().__init__()
        self.p = p
        self.drop = nn.Dropout1d(self.p)
    def forward(self,x):
        y = torch.permute(x,(0,2,1))
        y = self.drop(y)
        y = torch.permute(y,(0,2,1))
        return y

class _lindpout2d(nn.Module):
    def __init__(self,p:float = 0.2):
        super().__init__()
        self.p = p
        self.drop = nn.Dropout2d(self.p)
    def forward(self,x):
        y = torch.permute(x,(0,3,1,2))
        y = self.drop(y)
        y = torch.permute(y,(0,2,3,1))
        return y
    
class LRGenerator(nn.Module):
    def __init__(self, PatchSize, Headhdim,N,nchann,nchannout):
        super().__init__()
        self.rank = 3
        self.npatch = N//PatchSize
        self.psize = PatchSize
        self.nchout = nchannout
        self.nchin = nchann
        self.lrfeatbias = nn.Parameter(torch.zeros(1,nchannout//2,1,1))
        
        ## Layers
        self.vtokenproj = nn.Sequential(
            _lindpout3d(0.1),
            nn.Linear(PatchSize**2,2*PatchSize**2),
            nn.GELU(),
            _lindpout3d(0.1),
            #nn.Dropout3d(0.1),
            #nn.Linear(2*PatchSize**2,2*PatchSize**2),
            #nn.GELU(),
            nn.Linear(2*PatchSize**2,Headhdim)
        )
        self.htokenproj = nn.Sequential(  ## <- C x Np x Np x ps^2
            _lindpout3d(0.1),
            nn.Linear(PatchSize**2,2*PatchSize**2),
            nn.GELU(),
            _lindpout3d(0.1),
            #nn.Dropout3d(0.1),
            #nn.Linear(2*PatchSize**2,2*PatchSize**2),
            #nn.GELU(),
            nn.Linear(2*PatchSize**2,Headhdim)
        )#nn.Linear(PatchSize**2,Headhdim)
        self.Vproj = nn.Sequential(
            _lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            #nn.Dropout1d(0.1),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * N)
            ) #nn.Linear(Headhdim * self.npatch, self.rank * N)
        self.Hproj = nn.Sequential(
            _lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            _lindpout1d(0.1),
            #nn.Dropout1d(0.1),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * N)
            )#nn.Linear(Headhdim * self.npatch, self.rank * N)
        #self.Vprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N*self.rank) for i in range(self.rank)])
        #self.Hprojs = nn.ModuleList([nn.Linear(Headhdim*self.npatch,N*self.rank) for i in range(self.rank)])
        pe = get_2d_positional_encoding(N,N,nchann)
        pe = get_2d_positional_encoding(N, N, nchann)      # (N, N, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)              # (1, C, N, N)
        self.register_buffer("pos_enc", pe)
        self.vcln = nn.LayerNorm(Headhdim*self.npatch)
        self.hcln = nn.LayerNorm(Headhdim*self.npatch)
        
        self.vcompln = nn.LayerNorm(nchann//2)
        self.hcompln = nn.LayerNorm(nchann//2)
        
        nchout = self.nchout # nchann
        self.vchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//2),
            nn.GELU(),
            _lindpout2d(0.15),
            #nn.Dropout1d(0.15),
            nn.Linear(nchout//2,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            )##nn.Linear(nchann,nchann//2)
        self.hchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//2),
            nn.GELU(),
            _lindpout2d(0.15),
            #nn.Dropout1d(0.15),
            nn.Linear(nchout//2,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            ) #nn.Linear(nchann,nchann//2)
        self.fbn = nn.BatchNorm2d(nchann)
        #self.obn = nn.BatchNorm2d(nchannout//2)
    def forward(self, x):
        ## Patching and tokenization
        b,c,h,w = x.shape
        #print(x.shape)
        #print(self.pos_enc.shape)
        x = self.fbn(x)
        y = x + self.pos_enc # B x C x N x N
        p = self.psize
        np = self.npatch
        patches = y.unfold(2, p, p).unfold(3, p, p) ## B x Np x Np x Ps x Ps
        #patches = patches.contiguous().view(3, -1, p, p)
        patches = patches.contiguous().view(b,c,np,np,p*p) ## flattened patches
        
        ## Vertical / Horizontal tokens
        vtok = F.gelu(self.vtokenproj(patches))
        htok = F.gelu(self.htokenproj(patches))

        ### AGGREGATION
        # vertical component: for each column i, sum over rows j
        # vcol: (B,C,np,Headhdim)
        vcol = vtok.mean(dim=2)
        # horizontal component: for each row j, sum over cols i
        # hrow: (B,C,np,Headhdim)
        hrow = htok.mean(dim=3)

        # flatten np tokens into one vector per channel
        # vcomp/hcomp: (B,C,Headhdim*np)
        vcomp = vcol.reshape(b, c, -1)
        hcomp = hrow.reshape(b, c, -1)
        
        # vcomp : B x C x hdim*np
        #vcomp = self.vcln(vcomp)#.transpose(1,2)
        #hcomp = self.hcln(hcomp)#.transpose(1,2)
        
        #vcomp/hcomp : B x hdim*np x C
        #vcomp = F.gelu(self.vchannlin(vcomp).transpose(2,1))
        #hcomp = F.gelu(self.hchannlin(hcomp).transpose(2,1))
        
        # project to rank*N then reshape
        # V,H: (B,C,rank,N)
        V = self.Vproj(vcomp).view(b, self.nchin, self.rank, h)
        V = self.vchannlin(torch.permute(V,(0,2,3,1)))
        V = torch.permute(V,(0,3,1,2))
        V = F.gelu(V)
        
        Hm = self.Hproj(hcomp).view(b, self.nchin, self.rank, w)
        Hm = self.hchannlin(torch.permute(Hm,(0,2,3,1)))
        Hm = torch.permute(Hm,(0,3,1,2))
        Hm = F.gelu(Hm)

        # feat: sum_r V_r[:, :, :, i] * H_r[:, :, :, j]
        # -> (B,C,H,W)
        
        lrfeats = torch.einsum("bcrh,bcrw->bchw", V, Hm) + self.lrfeatbias 

        return lrfeats


class BTokenizer(nn.Module):
    def __init__(self,nblocks,nin,nout):
        super().__init__()
        ## nblock must be equal or greater than 2
        self.nblocks = nblocks
        self.bin = nn.Linear(nin,nout//2)
        self.bout = nn.Linear(nout//2,nout)
        self.ln = nn.LayerNorm(nout//2,elementwise_affine=False)
        self.blocks = nn.ModuleList([
            nn.Sequential(
            nn.Linear(nout//2,nout//2),
            nn.GELU()) for _ in range(nblocks-2)])
                
    def forward(self,x):
        y = 0.0
        y += self.bin(x)
        for block in self.blocks:
            ## pre-norm and add
            y += block(self.ln(y))
        return self.bout(y)
    
class LRGeneratorExp(nn.Module):
    def __init__(self, PatchSize, Headhdim,N,nchann,nchannout,down=False):
        super().__init__()
        self.rank = 3
        self.effN = N//2 if down  else N
        self.npatch = self.effN//PatchSize #if not down else self.effN//(PatchSize*2)
        self.psize = PatchSize
        self.nchout = nchannout
        self.nchin = nchann
        #self.lrfeatbias = nn.Parameter(torch.zeros(1,nchannout//2,1,1))
        BTSIZE = 8*Headhdim
        print("LR Generator patching")
        if down:
            self.inconv = nn.Conv2d(nchann,nchann,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    groups=nchann)
            #self.skip = nn.Conv2d(nchann,nchannout//2,
            #                        kernel_size=1,
            #                        stride=2,
            #                        bias=False)
        else:
            self.inconv = nn.Identity()
            #self.skip = nn.Identity()

        ## Layers
        self.perchanntok = nn.Conv2d(nchann,nchann,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False,
                                    groups=nchann)
        
        self.vtokenproj = nn.Sequential(
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,BTSIZE),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,Headhdim)
        )

        self.htokenproj = nn.Sequential(  ## <- C x Np x Np x ps^2
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,BTSIZE),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,Headhdim)
        )

        self.Vproj = nn.Sequential(
            _lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * self.effN),
            #_lindpout1d(0.1),
            #nn.GELU()
            ) #nn.Linear(Headhdim * self.npatch, self.rank * N)
        
        self.Hproj = nn.Sequential(
            _lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * self.effN),
            #_lindpout1d(0.1),
            #nn.GELU()
            )
        
        pe = get_2d_positional_encoding(self.effN, self.effN, nchann)      # (N, N, C)
        pe = pe.permute(2, 0, 1).unsqueeze(0)              # (1, C, N, N)
        self.register_buffer("pos_enc", pe)

        self.Btok = BTokenizer(nblocks=6,nin=self.psize**2,nout=BTSIZE)
        
        nchout = self.nchout # nchann
        self.vchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//4),
            nn.GELU(),
            _lindpout2d(0.15),
            nn.Linear(nchout//4,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            )##nn.Linear(nchann,nchann//2)
        self.hchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//4),
            nn.GELU(),
            _lindpout2d(0.15),
            nn.Linear(nchout//4,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            ) #nn.Linear(nchann,nchann//2)
        
        self.inbn = nn.BatchNorm2d(nchann)

    def forward(self, x):
        ## Patching and tokenization
        #y = self.inbn(x)
        #skip = self.skip(y)
        y = self.inconv(x)
        
        b,c,h,w = y.shape
        y = self.perchanntok(y)
        y = y + self.pos_enc # B x C x N x N
        p = self.psize
        np = self.npatch
        patches = y.unfold(2, p, p).unfold(3, p, p) ## B x Np x Np x Ps x Ps
        patches = patches.contiguous().view(b,c,np,np,p*p) ## flattened patches
        ## Raw Patch tokenizer
        tokens = self.Btok(patches) # B x Np x Np x 8Hdim
        
        ## Vertical / Horizontal tokens
        vtok = F.gelu(self.vtokenproj(tokens.mean(dim=2)))
        htok = F.gelu(self.htokenproj(tokens.mean(dim=3)))

        # flatten np tokens into one vector per channel
        # vcomp/hcomp: (B,C,Headhdim*np)
        vcomp = vtok.reshape(b, c, -1)
        hcomp = htok.reshape(b, c, -1)

        V = self.Vproj(vcomp).view(b, self.nchin, self.rank, h)
        V = self.vchannlin(torch.permute(V,(0,2,3,1)))
        V = torch.permute(V,(0,3,1,2))
        #V = F.gelu(V)
        
        Hm = self.Hproj(hcomp).view(b, self.nchin, self.rank, w)
        Hm = self.hchannlin(torch.permute(Hm,(0,2,3,1)))
        Hm = torch.permute(Hm,(0,3,1,2))
        #Hm = F.gelu(Hm)

        # feat: sum_r V_r[:, :, :, i] * H_r[:, :, :, j]
        # -> (B,C,H,W)
        
        lrfeats = torch.einsum("bcrh,bcrw->bchw", V, Hm) #+ self.lrfeatbias 

        return lrfeats#+skip

class LRGeneratorExp2(nn.Module):
    def __init__(self,ksize,stride, Headhdim,N,nchann,nchannout,down=False):
        super().__init__()
        self.rank = 3
        self.effN = N//2 if down  else N
        self.npatch = self.effN//stride
        self.nchout = nchannout
        self.nchin = nchann
        self.lrfeatbias = nn.Parameter(torch.zeros(1,nchannout//2,1,1))
        BTSIZE = 4*Headhdim
        
        pad = ksize // 2
        self.emb = nn.Sequential(
            nn.BatchNorm2d(nchann),
            nn.Conv2d(nchann, BTSIZE, kernel_size=ksize, stride=stride, padding=pad, bias=False),
            nn.GELU()
        )

        self.patcher = nn.Sequential(
            nn.BatchNorm2d(nchann),
            nn.Conv2d(nchann, nchann, kernel_size=ksize, stride=stride, padding=pad,
                      groups=nchann, bias=False)
        )
        
        if down:
            self.inconv = nn.Conv2d(nchann,nchann,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    groups=nchann)
        else:
            self.inconv = nn.Identity()

        ## Layers
        self.perchanntok = nn.Conv2d(nchann,nchann,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False,
                                    groups=nchann)
        
        self.vtokenproj = nn.Sequential(
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,Headhdim)
        )

        self.htokenproj = nn.Sequential(  ## <- C x Np x Np x ps^2
            _lindpout2d(0.1),
            nn.Linear(BTSIZE,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,BTSIZE//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(BTSIZE//2,Headhdim)
        )

        self.Vproj = nn.Sequential(
            #_lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * self.effN),
            _lindpout1d(0.1),
            #nn.GELU()
            ) #nn.Linear(Headhdim * self.npatch, self.rank * N)
        
        self.Hproj = nn.Sequential(
            #_lindpout1d(0.15),
            nn.Linear(Headhdim * self.npatch,Headhdim * self.npatch),
            nn.GELU(),
            _lindpout1d(0.1),
            nn.Linear(Headhdim * self.npatch,self.rank * self.effN),
            _lindpout1d(0.1),
            #nn.GELU()
            )
        
        nchout = self.nchout # nchann
        self.vchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(nchout//2,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            )##nn.Linear(nchann,nchann//2)
        self.hchannlin = nn.Sequential(
            nn.Linear(nchann,nchout//2),
            nn.GELU(),
            _lindpout2d(0.1),
            nn.Linear(nchout//2,nchout//2),
            #nn.GELU(),
            #nn.Linear(nchout//2,nchout//2)
            ) #nn.Linear(nchann,nchann//2)
        
        self.inbn = nn.BatchNorm2d(nchann)

    def forward(self, x):
        ## Patching and tokenization
        y = self.inbn(self.inconv(x))
        b,c,h,w = y.shape
        ## Raw Patch tokenizer
        emb = self.emb(y)              # B x tdim x Np x Np
        patcher = self.patcher(y)      # B x C x Np x Np

        emb = emb.permute(0, 2, 3, 1).unsqueeze(1)        # B x 1 x Np x Np x tdim
        patcher = patcher.unsqueeze(-1)                   # B x C x Np x Np x 1

        tokens = emb*(1.0 + patcher) 
        
        ## Vertical / Horizontal tokens
        vtok = F.gelu(self.vtokenproj(tokens.mean(dim=2)))
        htok = F.gelu(self.htokenproj(tokens.mean(dim=3)))

        # flatten np tokens into one vector per channel
        # vcomp/hcomp: (B,C,Headhdim*np)
        vcomp = vtok.reshape(b, c, -1)
        hcomp = htok.reshape(b, c, -1)

        V = self.Vproj(vcomp).view(b, self.nchin, self.rank, h)
        V = self.vchannlin(torch.permute(V,(0,2,3,1)))
        V = torch.permute(V,(0,3,1,2))
        #V = F.gelu(V)
        
        Hm = self.Hproj(hcomp).view(b, self.nchin, self.rank, w)
        Hm = self.hchannlin(torch.permute(Hm,(0,2,3,1)))
        Hm = torch.permute(Hm,(0,3,1,2))
        #Hm = F.gelu(Hm)

        # feat: sum_r V_r[:, :, :, i] * H_r[:, :, :, j]
        # -> (B,C,H,W)
        
        lrfeats = torch.einsum("bcrh,bcrw->bchw", V, Hm) + self.lrfeatbias 

        return lrfeats

class LRGeneratorConv(nn.Module):
    def __init__(self, PatchSize, Headhdim,N,nchann,nchannout):
        super().__init__()
        self.rank = 3
        self.npatch = N//PatchSize
        self.psize = PatchSize
        self.nchout = nchannout
        self.nchin = nchann
        self.lrfeatbias = nn.Parameter(torch.zeros(1,nchannout//2,1,1))
        self.tokwa = nn.Parameter(torch.ones(1,1,(N//4),1))
        self.tokwb = nn.Parameter(torch.ones(1,1,1,(N//4)))
        ## Tokenizer
        self.tdim = 32
        self.tokenizer = nn.Sequential(
            nn.BatchNorm2d(self.nchin),
            nn.Conv2d(in_channels=self.nchin,
                      out_channels=self.nchin,
                      kernel_size=5,
                      stride=4,
                      padding=2,
                      groups=self.nchin),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.nchin,
                out_channels=self.tdim,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        ) ## tdim x Np x Np, Np: N//4
         
        
        ## Layers
        self.channdec = nn.Sequential( # Channel component generator
            nn.Linear(self.tdim,self.tdim*2),
            _lindpout1d(0.15),
            nn.GELU(),
            nn.Linear(self.tdim*2,self.nchout//2),
            nn.GELU(),
        )
        self.hdec = nn.Sequential( # Horizontal component generator
            nn.Linear(self.tdim,self.tdim*2),
            _lindpout1d(0.15),
            nn.GELU(),
            #nn.Linear(self.tdim*2,N*self.rank),
            nn.Linear(self.tdim*2,N),
            nn.GELU()
        )
        self.vdec = nn.Sequential( # Vertical component generator
            nn.Linear(self.tdim,self.tdim*2),
            _lindpout1d(0.15),
            nn.GELU(),
            #nn.Linear(self.tdim*2,N*self.rank),
            nn.Linear(self.tdim*2,N),
            nn.GELU()
        )
        pe = get_1d_positional_encoding((N//4)**2,self.tdim).unsqueeze(0)# (1, L, tdim)
        #pe = pe.permute(1, 0).unsq(ueeze(0)              # (1, C, L)
        
        self.rankcompa = nn.Sequential(
            _lindpout1d(0.1),
            nn.Linear(N//4,self.rank**2)
        )
        
        self.rankcompb = nn.Sequential(
            _lindpout1d(0.1),
            nn.Linear(N//4,self.rank**2)
        )
        
        self.rankcompc = nn.Sequential(
            _lindpout1d(0.1),
            nn.Linear(N//4,self.rank)
        )
        
        self.register_buffer("pos_enc", pe)
        
    def forward(self, x: torch.Tensor):
        _,_,h,w = x.shape
        tok = self.tokenizer(x)
        b,c,np,_ = tok.shape 
        tok = tok.view(b,self.tdim,np**2)## tdim "basis" 
        tok = torch.permute(tok,(0,2,1)) + self.pos_enc
        tok = torch.permute(tok,(0,2,1))
        tok = tok.view(b,self.tdim,np,np)
        toka = (tok*self.tokwa).mean(dim=2) # B x tdim x np   
        tokb = (tok*self.tokwb).mean(dim=3) # B x tdim x np
        tokr = self.rankcompa(toka) + self.rankcompb(tokb) #B x tdim x rank^2
        tokr = torch.permute(tokr,(0,2,1)) # B x rank^2 x tdim 
        
        tokrc = self.rankcompc(toka+tokb) 
        tokrc = torch.permute(tokrc,(0,2,1))# B x rank x tdim 
        
        
        ## Vcomps: one per rank comp
        #vcomps = self.vdec(tok).mean(dim=1)
        vcomps = self.vdec(tokr)# 
        vcomps = torch.permute(vcomps,(0,2,1)) #B x tdim x  ~好啊
        vcomps = vcomps.view(b,h,self.rank,self.rank) # B x h x rank x rank
        
        ## Hcomps: one per rank comp
        #hcomps = self.hdec(tok).mean(dim=1)# Simple averaging
        hcomps = self.hdec(tokr)# B x tdim x h.rank
        hcomps = torch.permute(hcomps,(0,2,1))  
        hcomps = hcomps.view(b,h,self.rank,self.rank) # B x h x rank x rank
        
        ## Chcomps: one per rank comp
        #chcomps = self.channdec(tok).mean(dim=1) # B x Cout <- averaging
        chcomps = self.channdec(tokrc) # B x rank x Cout
        chcomps = torch.permute(chcomps,(0,2,1)) # B x Cout x r 
        #chcomps = chcomps.unsqueeze(-1).unsqueeze(-1)
        
        #lrfeats = torch.einsum("bhr,bwr->bhw", vcomps, hcomps)
        lrfeats = torch.einsum("bhrp,bwrp->bhwr", vcomps, hcomps)
        lrfeats = torch.einsum("bhwr,bcr->bchw", lrfeats,chcomps)
        
        #print(lrfeats.shape,chcomps.shape) 
        #lrfeats = chcomps*lrfeats.unsqueeze(1) + self.lrfeatbias
        return lrfeats
    """
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
        vtok = self.vtokenproj(patches)
        htok = self.htokenproj(patches)

        # vertical component: for each column i, sum over rows j
        # vcol: (B,C,np,Headhdim)
        vcol = vtok.sum(dim=2)

        # horizontal component: for each row j, sum over cols i
        # hrow: (B,C,np,Headhdim)
        hrow = htok.sum(dim=3)

        # flatten np tokens into one vector per channel
        # vcomp/hcomp: (B,C,Headhdim*np)
        vcomp = vcol.reshape(B, C, -1)
        hcomp = hrow.reshape(B, C, -1)

        # project to rank*N then reshape
        # V,H: (B,C,rank,N)
        V = self.Vproj(vcomp).view(B, C, self.rank, H)
        Hm = self.Hproj(hcomp).view(B, C, self.rank, W)

        # feat: sum_r V_r[:, :, :, i] * H_r[:, :, :, j]
        # -> (B,C,H,W)
        lrfeats = torch.einsum("bcrh,bcrw->bchw", V, Hm)

        return lrfeats
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
        """


        

class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1,Ni=None,isout=True,**kwargs):
        super().__init__()
        self.ni = Ni
        self.isout = isout
        self.bn1 = nn.BatchNorm2d(in_planes)
        #self.conv1 = nn.Conv2d(in_planes, in_planes//2, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes//2, kernel_size=1, bias=True)
        
        #self.bn2 = nn.BatchNorm2d(in_planes//2)
        self.bn2 = nn.BatchNorm2d(out_planes//2)
        #self.conv2 = nn.Conv2d(in_planes//2, in_planes//2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=stride, padding=1, bias=True)

        #self.bn3 = nn.BatchNorm2d(in_planes//2)
        self.bn3 = nn.BatchNorm2d(out_planes//2)
        #self.conv3 = nn.Conv2d(in_planes//2, out_planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(out_planes//2, out_planes, kernel_size=1, bias=True)

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
        out2 = F.interpolate(out, size=(self.ni,self.ni), mode='bilinear', align_corners=False)
        return (out,out2) if self.isout else out


class PreActBottleneckLR(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1,use_lr=False,N=32,Ni=32,isout=True):
        super().__init__()
        self.ni = Ni
        self.n=N
        self.isout = isout
        self.lrgen = LRGeneratorExp(PatchSize=4,
                                    Headhdim=4,
                                    N=N,
                                    nchann=in_planes,
                                    nchannout = out_planes) if use_lr else None
        if use_lr:
            print("Using LR module")
        self.convln = nn.BatchNorm2d(out_planes)
        self.convlr = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.lrbn = nn.BatchNorm2d(in_planes)

        self.convlrmatch = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes//2, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(out_planes//2)
        self.conv2 = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_planes//2)
        self.conv3 = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=1, bias=True)
        self.stride = stride
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            # In pre-act ResNet, shortcut conv happens on the *activated* input
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        outbn = 1.0*out#self.lrbn(x)#
        #shortcut = self.shortcut(out) if self.shortcut is not None else x
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        if self.lrgen is not None:
            lrfeats = self.lrgen(outbn)
            #lrfeats = self.convlr(lrfeats)
            if self.stride == 2:
                lrfeats = F.avg_pool2d(lrfeats, kernel_size=2, stride=2)
        else:
            lrfeats = self.convlrmatch(out) # self.convlr(out)

        out = torch.concat([out,lrfeats],dim=1)
        #out = out.permute(0,2,3,1)
        #out = self.convln(out)
        out = self.convlr(out)
        #out = out.permute(0,3,1,2)
        out = F.relu(out + shortcut)
        out2 = F.interpolate(out, size=(self.ni,self.ni), mode='bilinear', align_corners=False)
        return (out,out2) if self.isout else out
    
class PreActBottleneckLRAdapt(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1,use_lr=False,N=32,Ni=32,isout=True):
        super().__init__()
        self.ni = Ni
        self.n=N
        self.isout = isout
        self.lrgen = LRGeneratorExp(PatchSize=4,
                                    Headhdim=4,
                                    N=N,
                                    nchann=in_planes,
                                    nchannout = out_planes) if use_lr else None
        if use_lr:
            print("Using LR module")
        self.convln = nn.BatchNorm2d(out_planes)
        self.convlr = nn.Conv2d(out_planes, out_planes, kernel_size=1, bias=False)
        self.lrbn = nn.BatchNorm2d(in_planes)

        self.convlrmatch = nn.Conv2d(out_planes//2, out_planes//2, kernel_size=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes//8, kernel_size=1, bias=True)

        self.bn2 = nn.BatchNorm2d(out_planes//8)
        self.conv2 = nn.Conv2d(out_planes//8, out_planes//4, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_planes//4)
        self.conv3 = nn.Conv2d(out_planes//4, out_planes//2, kernel_size=1, bias=True)
        self.stride = stride
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            # In pre-act ResNet, shortcut conv happens on the *activated* input
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        xi = F.interpolate(x, size=(self.n,self.n), mode='bilinear', align_corners=False)
        out = F.relu(self.bn1(xi))
        outbn = 1.0*out#self.lrbn(x)#
        #shortcut = self.shortcut(out) if self.shortcut is not None else x
        shortcut = self.shortcut(out) if self.shortcut is not None else xi
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        if self.lrgen is not None:
            lrfeats = self.lrgen(outbn)
            #lrfeats = self.convlr(lrfeats)
            if self.stride == 2:
                lrfeats = F.avg_pool2d(lrfeats, kernel_size=2, stride=2)
        else:
            lrfeats = self.convlrmatch(out) # self.convlr(out)

        out = torch.concat([out,lrfeats],dim=1)
        out = self.convlr(out)
        out = F.relu(out + shortcut)
        out2 = F.interpolate(out, size=(self.ni,self.ni), mode='bilinear', align_corners=False)
        return out2#(out,out2) if self.isout else out

class PreActResNetExp(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,64,  num_blocks[0], stride=1,N=64,Ni=56) #512
        self.layer2 = self._make_layer(block, 64,128, num_blocks[1], stride=2,N=64,Ni=28) #128
        self.layer3 = self._make_layer(block, 128,256, num_blocks[2], stride=2,N=32,Ni=14) # 64
        self.layer4 = self._make_layer(block, 256,512, num_blocks[3], stride=2,N=16,Ni=7) #32
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, iplanes, oplanes, nblocks, stride, N,Ni):
        strides = [stride] + [1] * (nblocks - 1)
        layers = []
        in_ch = iplanes
        curN = N

        for i, s in enumerate(strides):
            layers.append(block(
                in_ch, oplanes,
                stride=s,
                use_lr=(i == 0),
                N=curN,
                Ni=Ni,
                isout=(i==(len(strides)-1))
            ))
            in_ch = oplanes
            if s == 2:
                curN = curN // 2  # spatial downsample happens here
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out,_ = self.layer1(out)
        out,_ = self.layer2(out)
        out,_ = self.layer3(out)
        out,_ = self.layer4(out)

        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

class PreActResNetAdapt(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64

        #self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ### Layers from ResNet18
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        
        self.conv1 = copy.deepcopy(resnet.conv1)
        self.bn1 = copy.deepcopy(resnet.bn1)
        self.relu = copy.deepcopy(resnet.relu)
        self.maxpool = copy.deepcopy(resnet.maxpool)
        
        self.layer11 = copy.deepcopy(resnet.layer1[0])
        self.layer12 = copy.deepcopy(resnet.layer1[1])
        
        self.layer21 = copy.deepcopy(resnet.layer2[0])
        self.layer22 = copy.deepcopy(resnet.layer2[1])
        
        self.layer31 = copy.deepcopy(resnet.layer3[0])
        self.layer32 = copy.deepcopy(resnet.layer3[1])
        
        self.layer41 = copy.deepcopy(resnet.layer4[0])
        self.layer42 = copy.deepcopy(resnet.layer4[1])
        
        
        ###
        self.convl11 = self._make_layer(block, 64,64,1, stride=1,N=64,Ni=56)
        self.convl12 = self._make_layer(block, 64,64,1, stride=1,N=64,Ni=56)
        
        self.convl21 = self._make_layer(block, 64,128, 1, stride=2,N=64,Ni=28) #128
        self.convl22 = self._make_layer(block, 128,128, 1, stride=1,N=64,Ni=28) #128
        
        self.convl31 = self._make_layer(block, 128,256,1, stride=2,N=32,Ni=14) # 64
        self.convl32 = self._make_layer(block, 256,256,1, stride=1,N=32,Ni=14) # 64
        
        self.convl41 = self._make_layer(block, 256,512, 1, stride=2,N=16,Ni=7) #32
        self.convl42 = self._make_layer(block, 512,512, 1, stride=1,N=16,Ni=7) #32
        
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.freeze_backbone()
    def freeze_backbone(self):
        modules = [
            self.conv1, self.bn1,
            self.layer11, self.layer12,
            self.layer21, self.layer22,
            self.layer31, self.layer32,
            self.layer41, self.layer42
        ]

        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
    def train(self, mode=True):
        super().train(mode)
        self.freeze_backbone()
        return self
    
    def _make_layer(self, block, iplanes, oplanes, nblocks, stride, N,Ni):
        strides = [stride] + [1] * (nblocks - 1)
        layers = []
        in_ch = iplanes
        curN = N

        for i, s in enumerate(strides):
            layers.append(block(
                in_ch, oplanes,
                stride=s,
                use_lr=(i == 0),
                N=curN,
                Ni=Ni,
                isout=(i==(len(strides)-1))
            ))
            in_ch = oplanes
            if s == 2:
                curN = curN // 2  # spatial downsample happens here
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        outs = self.maxpool(out)
        
        ## 1
        outs = self.layer11(outs) + self.convl11(outs)
        outs = self.layer12(outs) + self.convl12(outs)
        
        ## 2
        outs = self.layer21(outs) + self.convl21(outs)
        outs = self.layer22(outs) + self.convl22(outs)
        
        ## 3
        outs = self.layer31(outs) + self.convl31(outs)
        outs = self.layer32(outs) + self.convl32(outs)
        
        ## 4
        outs = self.layer41(outs) + self.convl41(outs)
        outs = self.layer42(outs) + self.convl42(outs)
        
        out = F.relu(self.bn_final(outs))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

class _resnetBlockLR(nn.Module):
    def __init__(self,cin,cout,N,psize,hdim,linear=True,use_lr=True):
        super().__init__()
        self.use_lr = use_lr
        self.conv11 = nn.Conv2d(cin,cout//2, kernel_size=3, stride=2,padding=1,bias=False)
        self.bn11 = nn.BatchNorm2d(cout//2)
        self.conv12 = nn.Conv2d(cout//2,cout//2, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn12 = nn.BatchNorm2d(cout//2)
        self.skip1 = nn.Sequential(
                nn.Conv2d(cin,cout, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(cout))
        #ksize,stride, Headhdim,N,nchann,nchannout,down=False
        if use_lr:
            self.lr1 = LRGeneratorExp(psize,hdim,N=N,nchann=cin,nchannout=cout,down=True)
        #self.lr1 = LRGeneratorExp2(3,2,hdim,N=N,nchann=cin,nchannout=cout,down=True)
        self.bno1 = nn.BatchNorm2d(cout)
        if use_lr:
            self.outlin1 = nn.Conv2d(cout,cout, kernel_size=1, stride=1,bias=False)
        else:
            self.outlin1 = nn.Conv2d(cout//2,cout, kernel_size=1, stride=1,bias=False)
        
        self.conv21 = nn.Conv2d(cout,cout//2, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn21 = nn.BatchNorm2d(cout//2)
        self.conv22 = nn.Conv2d(cout//2,cout//2, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn22 = nn.BatchNorm2d(cout//2)
        if use_lr:
            self.lr2 = LRGeneratorExp(psize,hdim,N=N//2,nchann=cout,nchannout=cout)
        #self.lr2 = LRGeneratorExp2(3,2,hdim,N=N//2,nchann=cout,nchannout=cout)
        self.bno2 = nn.BatchNorm2d(cout)
        if use_lr:
            self.outlin2 = nn.Conv2d(cout,cout, kernel_size=1, stride=1,bias=False)
        else:
            self.outlin2 = nn.Conv2d(cout//2,cout, kernel_size=1, stride=1,bias=False)
    def _forwardlr(self,x):
        skip = self.skip1(x)
        lr = self.lr1(x)
        out = F.relu(self.bn11(self.conv11(x)))
        out = self.bn12(self.conv12(out))
        out = F.relu(self.outlin1(self.bno1(torch.concat([out,lr],dim=1)))+skip)
        skip = 1.0*out
        lr = self.lr2(out)
        out = F.relu(self.bn21(self.conv21(out)))
        out = self.bn22(self.conv22(out))
        out = F.relu(self.outlin2(self.bno2(torch.concat([out,lr],dim=1)))+skip)
        return out
    
    def _forwardconv(self,x):
        skip = self.skip1(x)
        out = F.relu(self.bn11(self.conv11(x)))
        out = self.bn12(self.conv12(out))
        out = F.relu(self.outlin1(out)+skip)
        skip = 1.0*out
        out = F.relu(self.bn21(self.conv21(out)))
        out = self.bn22(self.conv22(out))
        out = F.relu(self.outlin2(out)+skip)
        return out
    def forward(self,x):
        if self.use_lr:
            return self._forwardlr(x)
        else:
            return self._forwardconv(x)

    
class _resConvBlock(nn.Module):
    def __init__(self,layer,bidx,cin,cout,linear=True):
        super().__init__()
        self.conv1 = copy.deepcopy(layer[bidx].conv1)
        self.bn1 = copy.deepcopy(layer[bidx].bn1)
        self.relu = copy.deepcopy(layer[bidx].relu)
        self.conv2 = copy.deepcopy(layer[bidx].conv2)
        self.bn2 = copy.deepcopy(layer[bidx].bn2)
        if linear:
            self.outlin = nn.Conv2d(cin,cout//2, kernel_size=1, stride=1, bias=False)
        else:
            self.outlin = nn.Identity()
        self.freeze()
    def freeze(self):
        modules = [
            self.conv1,
            self.bn1,
            self.conv2,
            self.bn2
        ]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

    def forward(self,x):
        with torch.no_grad():
            out = self.conv1(x)
            out = F.relu(self.bn1(out))
            out = self.conv2(out)
            out = self.bn2(out)
        out = self.outlin(out)
        return out
    
    def train(self, mode=True):
        super().train(mode)
        self.freeze()
        return self

class ResNetPartialSingle(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64

        ### Layers from pre-trained ResNet18
        self.frozmodel = models.resnet18(weights="IMAGENET1K_V1")
        model = ResNetOriginal(None,None)
        ckpt = torch.load("pre-trained/resnet18vanilla.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        ### Copy the stem for 256x256 inputs
        self.conv1 = copy.deepcopy(self.frozmodel.conv1)
        self.bn1 = copy.deepcopy(self.frozmodel.bn1)
        self.relu = copy.deepcopy(self.frozmodel.relu)
        self.maxpool = copy.deepcopy(self.frozmodel.maxpool)
        self.fc = copy.deepcopy(model.fc)
        
        ## ConvBlocks
        self.conv11 = _resConvBlock(self.frozmodel.layer1,0,64,64)
        self.conv12 = _resConvBlock(self.frozmodel.layer1,1,64,64)
        self.conv21 = _resConvBlock(self.frozmodel.layer2,0,128,128)
        self.conv22 = _resConvBlock(self.frozmodel.layer2,1,128,128)
        self.conv31 = _resConvBlock(self.frozmodel.layer3,0,256,256)
        self.conv32 = _resConvBlock(self.frozmodel.layer3,1,256,256)
        self.conv41 = _resConvBlock(self.frozmodel.layer4,0,512,512)
        self.conv42 = _resConvBlock(self.frozmodel.layer4,1,512,512)

        ### LR modules
        self.lr11 = LRGeneratorExp2(5,4,4,N=64,nchann=64,nchannout=64)
        self.lr12 = LRGeneratorExp2(5,4,4,N=64,nchann=64,nchannout=64)
        
        self.lr21 = LRGeneratorExp2(5,4,4,N=64,nchann=64,nchannout=128,down=True) #128
        self.lr22 = LRGeneratorExp2(5,4,4,N=32,nchann=128,nchannout=128) #128
        
        self.lr31 = LRGeneratorExp2(3,2,4,N=32,nchann=128,nchannout=256,down=True) # 64
        self.lr32 = LRGeneratorExp2(3,2,4,N=16,nchann=256,nchannout=256) # 64
        
        self.lr41 = LRGeneratorExp2(3,2,4,N=16,nchann=256,nchannout=512,down=True) #32
        self.lr42 = LRGeneratorExp2(3,2,4,N=8,nchann=512,nchannout=512) #32
        
        ### Skips
        self.skip2 = copy.deepcopy(self.frozmodel.layer2[0].downsample)
        self.skip3 = copy.deepcopy(self.frozmodel.layer3[0].downsample)
        self.skip4 = copy.deepcopy(self.frozmodel.layer4[0].downsample)
        """
        self.skip2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128))
        self.skip3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256))
        self.skip4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512))
        """
        #self.bn_final = nn.BatchNorm2d(512)
        self.fc_diff = nn.Linear(512, num_classes)
        self.freeze()
    
    def freeze(self):
        modules = [
            self.conv1,
            self.bn1,
            self.skip2,
            self.skip3,
            self.skip4,
            self.fc,
            self.frozmodel,
        ]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            
    def train(self, mode=True):
        super().train(mode)
        self.freeze()
        return self

    def forward(self, x):
        def mixfeats(inx,lr,conv,skip):
            rnf = conv(inx)
            lrf = lr(inx)
            out = torch.concat([rnf,lrf],dim=1)
            return F.relu(out+skip) #F.layer_norm(out, normalized_shape, weight=None, bias=None, eps=1e-05)
        
        ## Stem@256x256
        outlr = self.relu(self.bn1(self.conv1(x)))
        outlr = self.maxpool(outlr)
        
        ## 1
        outlr = mixfeats(outlr,self.lr11,self.conv11,outlr)
        outlr = mixfeats(outlr,self.lr12,self.conv12,outlr)
        
        ## 2
        skip = self.skip2(outlr)
        outlr = mixfeats(outlr,self.lr21,self.conv21,skip)
        outlr = mixfeats(outlr,self.lr22,self.conv22,outlr)
        
        ## 3
        skip = self.skip3(outlr)
        outlr = mixfeats(outlr,self.lr31,self.conv31,skip)
        outlr = mixfeats(outlr,self.lr32,self.conv32,outlr)
        
        ## 4
        skip = self.skip4(outlr)
        outlr = mixfeats(outlr,self.lr41,self.conv41,skip)
        outlr = mixfeats(outlr,self.lr42,self.conv42,outlr)
        
        #out = F.relu(self.bn_final(outlr))
        out = F.adaptive_avg_pool2d(outlr, 1).flatten(1)
        out = self.fc(out)+0.3*self.fc_diff(out)
        return out

class ResNetPartialSequential(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        ### Layers from pre-trained ResNet18
        self.frozmodel = models.resnet18(weights="IMAGENET1K_V1")
        model = ResNetOriginal(None,None)
        ckpt = torch.load("pre-trained/resnet18vanilla.pt", map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()

        ### Copy the stem for 256x256 inputs
        self.conv1 = copy.deepcopy(self.frozmodel.conv1)
        self.bn1 = copy.deepcopy(self.frozmodel.bn1)
        self.relu = copy.deepcopy(self.frozmodel.relu)
        self.maxpool = copy.deepcopy(self.frozmodel.maxpool)
        self.fc = copy.deepcopy(model.fc)
        
        ### Linear inputs
        self.lin11 =  nn.Sequential(
                    nn.Conv2d(64,64, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(64),
                    )
        self.lin12 =  nn.Sequential(
                    nn.Conv2d(64,64, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(64),
                    )
        self.lin21 =  nn.Sequential(
                    nn.Conv2d(128,128, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(128),
                    )
        self.lin22 =  nn.Sequential(
                    nn.Conv2d(128,128, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(128),
                    )
        self.lin31 =  nn.Sequential(
                    nn.Conv2d(256,256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256),
                    )
        self.lin32 =  nn.Sequential(
                    nn.Conv2d(256,256, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(256),
                    )
        self.lin41 =  nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(512),
                    )
        self.lin42 =  nn.Sequential(
                    nn.Conv2d(512,512, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(512),
                    )
        
        ## ConvBlocks
        self.conv11 = _resConvBlock(self.frozmodel.layer1,0,64,64,False)
        self.conv12 = _resConvBlock(self.frozmodel.layer1,1,64,64,False)
        self.conv21 = _resConvBlock(self.frozmodel.layer2,0,128,128,False)
        self.conv22 = _resConvBlock(self.frozmodel.layer2,1,128,128,False)
        self.conv31 = _resConvBlock(self.frozmodel.layer3,0,256,256,False)
        self.conv32 = _resConvBlock(self.frozmodel.layer3,1,256,256,False)
        self.conv41 = _resConvBlock(self.frozmodel.layer4,0,512,512,False)
        self.conv42 = _resConvBlock(self.frozmodel.layer4,1,512,512,False)
        
        ### LR modules
        self.lr11 = LRGeneratorExp(8,4,N=64,nchann=64,nchannout=64*2)
        self.lr12 = LRGeneratorExp(8,4,N=64,nchann=64,nchannout=64*2)
        
        #self.lr21 = LRGeneratorExp2(5,4,4,N=32,nchann=64*2,nchannout=128*2,down=True) #128
        #self.lr22 = LRGeneratorExp2(5,4,4,N=32,nchann=128,nchannout=128*2) #128
        
        self.lr21 = LRGeneratorExp(4,4,N=64,nchann=64,nchannout=128*2,down=True) #128
        self.lr22 = LRGeneratorExp(4,4,N=32,nchann=128,nchannout=128*2) #128
        
        #self.lr31 = LRGeneratorExp2(3,2,8,N=16,nchann=128*2,nchannout=256*2,down=True) # 64
        #self.lr32 = LRGeneratorExp2(3,2,8,N=16,nchann=256,nchannout=256*2) # 64
        
        self.lr31 = LRGeneratorExp(2,4,N=32,nchann=128,nchannout=256*2,down=True) # 64
        self.lr32 = LRGeneratorExp(2,4,N=16,nchann=256,nchannout=256*2) # 64
        
        #self.lr41 = LRGeneratorExp2(3,2,8,N=8,nchann=256*2,nchannout=512*2,down=True) #32
        #self.lr42 = LRGeneratorExp2(3,2,8,N=8,nchann=512,nchannout=512*2) #32
        
        self.lr41 = LRGeneratorExp(2,4,N=16,nchann=256,nchannout=512*2,down=True) #32
        self.lr42 = LRGeneratorExp(2,4,N=8,nchann=512,nchannout=512*2) #32
        
        ### Skips
        """
        self.skip2ft = nn.Sequential(
                nn.Conv2d(64, 128//4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128//4),
                nn.Conv2d(128//4, 128, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(128))
        self.skip3ft = nn.Sequential(
                nn.Conv2d(128, 256//4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256//4),
                nn.Conv2d(256//4, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(256))
        self.skip4ft = nn.Sequential(
                nn.Conv2d(256, 512//4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512//4),
                nn.Conv2d(512//4, 512, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(512))
        """

        self.skip2 = copy.deepcopy(self.frozmodel.layer2[0].downsample)
        self.skip3 = copy.deepcopy(self.frozmodel.layer3[0].downsample)
        self.skip4 = copy.deepcopy(self.frozmodel.layer4[0].downsample)
        
        #self.bn_final = nn.BatchNorm2d(512)
        self.fc_diff = nn.Sequential(
            #nn.Linear(512, num_classes//5),
            #nn.GELU(),
            nn.Linear(512, num_classes)
            )
        self.freeze()
    
    def freeze(self):
        modules = [
            self.conv1,
            self.bn1,
            self.skip2,
            self.skip3,
            self.skip4,
            self.conv11,
            self.conv12,
            self.conv21,
            self.conv22,
            self.conv31,
            self.conv32,
            self.conv41,
            self.conv42,
            #self.fc,
            #self.frozmodel,
        ]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            
    def train(self, mode=True):
        super().train(mode)
        self.freeze()
        return self

    def forward(self, x):
        def mixfeats(inx,lr,conv,lin,skip):
            rnf = lin(F.relu(conv(inx)+skip))
            #lrf = lr(rnf)
            lrf = lr(inx)
            out = rnf+lrf
            return F.relu(out)#0.7*F.relu(rnf+skip)+0.3*F.relu(lrf) #F.layer_norm(out, normalized_shape, weight=None, bias=None, eps=1e-05)
        
        ## Stem@256x256
        outlr = self.relu(self.bn1(self.conv1(x)))
        outlr = self.maxpool(outlr)
        
        ## 1
        outlr = mixfeats(outlr,self.lr11,self.conv11,self.lin11,outlr)
        outlr = mixfeats(outlr,self.lr12,self.conv12,self.lin12,outlr)
        
        ## 2
        skip = self.skip2(outlr)#+self.skip2ft(outlr)
        outlr = mixfeats(outlr,self.lr21,self.conv21,self.lin21,skip)
        outlr = mixfeats(outlr,self.lr22,self.conv22,self.lin22,outlr)
        
        ## 3
        skip = self.skip3(outlr)#+self.skip3ft(outlr)
        outlr = mixfeats(outlr,self.lr31,self.conv31,self.lin31,skip)
        outlr = mixfeats(outlr,self.lr32,self.conv32,self.lin32,outlr)
        
        ## 4
        skip = self.skip4(outlr)#+self.skip4ft(outlr)
        outlr = mixfeats(outlr,self.lr41,self.conv41,self.lin41,skip)
        outlr = mixfeats(outlr,self.lr42,self.conv42,self.lin42,outlr)
        
        #out = F.relu(self.bn_final(outlr))
        out = F.adaptive_avg_pool2d(outlr, 1).flatten(1)
        out = self.fc(out)#+0.1*self.fc_diff(out)
        return out


class ResNet2Back(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64
        
        ### Layers from pre-trained ResNet18
        self.frozmodel = models.resnet18(weights="IMAGENET1K_V1")
        self.fextr = create_feature_extractor(self.frozmodel,
                                              {"layer1.0.bn2":"rn11",
                                                "layer1.1.bn2":"rn12",
                                                "layer2.0.bn2":"rn21",
                                                "layer2.1.bn2":"rn22",
                                                "layer3.0.bn2":"rn31",
                                                "layer3.1.bn2":"rn32",
                                                "layer4.0.bn2":"rn41",
                                                "layer4.1.bn2":"rn42"
                                                })
        
        ### Copy the stem for 256x256 inputs
        self.conv1 = copy.deepcopy(self.frozmodel.conv1)
        self.bn1 = copy.deepcopy(self.frozmodel.bn1)
        self.relu = copy.deepcopy(self.frozmodel.relu)
        self.maxpool = copy.deepcopy(self.frozmodel.maxpool)
        
        ### LR modules
        self.lr11 = LRGeneratorExp(4,4,N=64,nchann=64,nchannout=64)
        self.lr12 = LRGeneratorExp(4,4,N=64,nchann=64,nchannout=64)
        
        self.lr21 = LRGeneratorExp(4,4,N=64,nchann=64,nchannout=128,down=True) #128
        self.lr22 = LRGeneratorExp(4,4,N=32,nchann=128,nchannout=128) #128
        
        self.lr31 = LRGeneratorExp(4,4,N=32,nchann=128,nchannout=256,down=True) # 64
        self.lr32 = LRGeneratorExp(4,4,N=16,nchann=256,nchannout=256) # 64
        
        self.lr41 = LRGeneratorExp(4,4,N=16,nchann=256,nchannout=512,down=True) #32
        self.lr42 = LRGeneratorExp(4,4,N=8,nchann=512,nchannout=512) #32
        
        ### Skips
        self.skip2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128))
        self.skip3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(256))
        self.skip4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(512))
        
        ### Linear projs
        self.lin11 = nn.Conv2d(64,32, kernel_size=1, stride=1, padding=0, bias=False)
        self.lin12 = nn.Conv2d(64,32, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin21 = nn.Conv2d(128,64, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin22 = nn.Conv2d(128,64, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin31 = nn.Conv2d(256,128, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin32 = nn.Conv2d(256,128, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin41 = nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0, bias=True)
        self.lin42 = nn.Conv2d(512,256, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)
        self.freeze_backbone()
    
    def freeze_backbone(self):
        modules = [
            self.conv1,
            self.bn1,
            self.frozmodel,
            self.fextr
        ]

        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            
    def train(self, mode=True):
        super().train(mode)
        self.freeze_backbone()
        return self

    def forward(self, x):
        def mixfeats(lrin,rnin,size,linlayer):
            rnf = linlayer(rnin)
            rnf = F.interpolate(rnf, size=(size,size), mode='bilinear', align_corners=False)
            out = torch.concat([lrin,rnf],dim=1)
            return out #F.layer_norm(out, normalized_shape, weight=None, bias=None, eps=1e-05)
        
        ## Feat extraction@224x224
        with torch.no_grad():
            xrn = F.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
            feats = self.fextr(xrn)
        ## Stem@256x256
        outlr = self.relu(self.bn1(self.conv1(x)))
        outlr = self.maxpool(outlr)
        
        ## 1
        outlr = F.relu(mixfeats(self.lr11(outlr),feats['rn11'],64,self.lin11)+outlr) ## N: 56->64 | 
        outlr = F.relu(mixfeats(self.lr12(outlr),feats['rn12'],64,self.lin12)+outlr)
        
        ## 2
        skip = self.skip2(outlr)
        outlr = F.relu(mixfeats(self.lr21(outlr),feats['rn21'],32,self.lin21)+skip) ## N: 28->32 | 
        outlr = F.relu(mixfeats(self.lr22(outlr),feats['rn22'],32,self.lin22)+outlr)
        
        ## 3
        skip = self.skip3(outlr)
        outlr = F.relu(mixfeats(self.lr31(outlr),feats['rn31'],16,self.lin31)+skip) ## N: 28->32 | 
        outlr = F.relu(mixfeats(self.lr32(outlr),feats['rn32'],16,self.lin32)+outlr)
        
        ## 4
        skip = self.skip4(outlr)
        outlr = F.relu(mixfeats(self.lr41(outlr),feats['rn41'],8,self.lin41)+skip) ## N: 28->32 | 
        outlr = F.relu(mixfeats(self.lr42(outlr),feats['rn42'],8,self.lin42)+outlr)
        
        out = F.relu(self.bn_final(outlr))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        out = self.fc(out)
        return out

class ResNetOriginal(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.frozmodel = models.resnet18(weights="IMAGENET1K_V1")
        ### Copy the stem for 256x256 inputs
        self.conv1 = copy.deepcopy(self.frozmodel.conv1)
        self.bn1 = copy.deepcopy(self.frozmodel.bn1)
        self.relu = copy.deepcopy(self.frozmodel.relu)
        self.maxpool = copy.deepcopy(self.frozmodel.maxpool)
        
        ## ConvBlocks
        self.conv11 = _resConvBlock(self.frozmodel.layer1,0,64,64,False)
        self.conv12 = _resConvBlock(self.frozmodel.layer1,1,64,64,False)
        self.conv21 = _resConvBlock(self.frozmodel.layer2,0,128,128,False)
        self.conv22 = _resConvBlock(self.frozmodel.layer2,1,128,128,False)
        self.conv31 = _resConvBlock(self.frozmodel.layer3,0,256,256,False)
        self.conv32 = _resConvBlock(self.frozmodel.layer3,1,256,256,False)
        self.conv41 = _resConvBlock(self.frozmodel.layer4,0,512,512,False)
        self.conv42 = _resConvBlock(self.frozmodel.layer4,1,512,512,False)
        
        ## Skip layers
        self.skip2 = copy.deepcopy(self.frozmodel.layer2[0].downsample)
        self.skip3 = copy.deepcopy(self.frozmodel.layer3[0].downsample)
        self.skip4 = copy.deepcopy(self.frozmodel.layer4[0].downsample)
        self.fc = nn.Linear(512, num_classes)
        self.freeze_backbone()
    
    def freeze_backbone(self):
        modules = [
            self.conv1,
            self.bn1,
            self.conv11,
            self.conv12,
            self.conv21,
            self.conv22,
            self.conv31,
            self.conv32,
            self.conv41,
            self.conv42,
            self.skip2,
            self.skip3,
            self.skip4
        ]

        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            
    def train(self, mode=True):
        super().train(mode)
        self.freeze_backbone()
        return self
    
    def forward(self,x):
        def addrelu(inx,conv,skip):
            out = conv(inx)
            return F.relu(out+skip)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = addrelu(out,self.conv11,out)
        out = addrelu(out,self.conv12,out)
        
        skip2 = self.skip2(out)
        out = addrelu(out,self.conv21,skip2)
        out = addrelu(out,self.conv22,out)
        
        skip3 = self.skip3(out)
        out = addrelu(out,self.conv31,skip3)
        out = addrelu(out,self.conv32,out)
        
        skip4 = self.skip4(out)
        out = addrelu(out,self.conv41,skip4)
        out = addrelu(out,self.conv42,out)
        
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)        
        out = self.fc(out)

        return out
    
class ResNetOriginalLR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3,use_lr=True):
        super().__init__()
        self.frozmodel = models.resnet18(weights="IMAGENET1K_V1")
        ### Copy the stem for 256x256 inputs
        self.conv1 = copy.deepcopy(self.frozmodel.conv1)
        self.bn1 = copy.deepcopy(self.frozmodel.bn1)
        self.relu = copy.deepcopy(self.frozmodel.relu)
        self.maxpool = copy.deepcopy(self.frozmodel.maxpool)
        
        ## ConvBlocks
        self.conv11 = _resConvBlock(self.frozmodel.layer1,0,64,64,False)
        self.conv12 = _resConvBlock(self.frozmodel.layer1,1,64,64,False)
        self.conv21 = _resConvBlock(self.frozmodel.layer2,0,128,128,False)
        self.conv22 = _resConvBlock(self.frozmodel.layer2,1,128,128,False)
        #self.conv31 = _resConvBlock(self.frozmodel.layer3,0,256,256,False)
        #self.conv32 = _resConvBlock(self.frozmodel.layer3,1,256,256,False)
        #self.conv41 = _resConvBlock(self.frozmodel.layer4,0,512,512,False)
        #self.conv42 = _resConvBlock(self.frozmodel.layer4,1,512,512,False)
        
        ## ResnetLR
        #self.reslr = _resnetBlockLR(cin=256,cout=512,N=16,psize=2,hdim=4)
        self.reslr = _resnetBlockLR(cin=128,cout=256,N=32,psize=4,hdim=16,use_lr=use_lr)
        self.reslr2 = _resnetBlockLR(cin=256,cout=512,N=16,psize=2,hdim=8,use_lr=use_lr)
        
        ## Skip layers
        self.skip2 = copy.deepcopy(self.frozmodel.layer2[0].downsample)
        #self.skip3 = copy.deepcopy(self.frozmodel.layer3[0].downsample)
        #self.skip4 = copy.deepcopy(self.frozmodel.layer4[0].downsample)
        self.fc = nn.Linear(512, num_classes)
        self.freeze_backbone()
    
    def freeze_backbone(self):
        modules = [
            self.conv1,
            self.bn1,
            self.conv11,
            self.conv12,
            self.conv21,
            self.conv22,
            #self.conv31,
            #self.conv32,
            #self.conv41,
            #self.conv42,
            self.skip2,
            #self.skip3,
            #self.skip4,
        ]

        for m in modules:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()
            
    def train(self, mode=True):
        super().train(mode)
        self.freeze_backbone()
        return self
    
    def forward(self,x):
        def addrelu(inx,conv,skip):
            out = conv(inx)
            return F.relu(out+skip)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = addrelu(out,self.conv11,out)
        out = addrelu(out,self.conv12,out)
        
        skip2 = self.skip2(out)
        out = addrelu(out,self.conv21,skip2)
        out = addrelu(out,self.conv22,out)
        
        #skip3 = self.skip3(out)
        #out = addrelu(out,self.conv31,skip3)
        #out = addrelu(out,self.conv32,out)
        
        out = self.reslr(out)
        out = self.reslr2(out)
        #skip4 = self.skip4(out)
        #out = addrelu(out,self.conv41,skip4)
        #out = addrelu(out,self.conv42,out)
        #out = self.reslr(out)
        
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)        
        out = self.fc(out)

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, in_ch=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,64,  num_blocks[0], stride=1,N=64,Ni=56) #512
        self.layer2 = self._make_layer(block, 64,128, num_blocks[1], stride=2,N=64,Ni=28) #128
        self.layer3 = self._make_layer(block, 128,256, num_blocks[2], stride=2,N=32,Ni=14) # 64
        self.layer4 = self._make_layer(block, 256,512, num_blocks[3], stride=2,N=16,Ni=7) #32
        self.bn_final = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, num_classes)

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


def preact_resnet18_bottleneck(num_classes=1000, in_ch=3,nblocks=1,useLR=True):
    # "ResNet-18 depth schedule" but bottleneck blocks
    #block_type = PreActBottleneckLRAdapt if useLR else PreActBottleneck
    return ResNetOriginalLR(block_type,
                        [nblocks, nblocks, nblocks, nblocks],
                        num_classes=num_classes,
                        in_ch=in_ch,
                        use_lr=useLR) 