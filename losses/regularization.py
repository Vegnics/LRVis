import torch
import torch.nn.functional as F
import sys,os

sys.path.insert(0,os.getcwd()) 
if "../" not in sys.path:
    sys.path.insert(0,"../")
from modules.lowrank2 import LRGeneratorExp,LRGeneratorExp2

def orth_loss_factors(Factor):
    """
    Factor: (B, C, R, S)
    Enforce orthogonality across rank dimension R.
    """
    B, C, R, S = Factor.shape

    # normalize each rank vector along spatial dimension
    X = F.normalize(Factor, p=2, dim=-1)          # (B,C,R,S)

    # Gram matrix over rank components
    G = torch.matmul(X, X.transpose(-1, -2))      # (B,C,R,R)

    I = torch.eye(R, device=Factor.device, dtype=Factor.dtype)
    I = I.view(1, 1, R, R)

    return ((G - I) ** 2).mean()

def model_orth_loss(model):
    losses = []
    for m in model.modules():
        if isinstance(m, LRGeneratorExp2):
            if hasattr(m, "last_V") and hasattr(m, "last_H"):
                losses.append(orth_loss_factors(m.last_V))
                losses.append(orth_loss_factors(m.last_H))
    if len(losses) == 0:
        print("NO V H")
        return 0.0
    loss = sum(losses) / len(losses)
    #print(type(loss),loss)
    return loss