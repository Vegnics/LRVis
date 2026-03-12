from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
import torchvision.io
import torchvision.transforms.functional as TF

import torch
import sys
import os
from matplotlib import pyplot as plt

sys.path.insert(0,os.getcwd()) 
if "../" not in sys.path:
    sys.path.insert(0,"../")
print(sys.path,os.getcwd())


def pca(X, k):
    """
    X: (N, D) feature matrix
    k: number of principal components
    """

    # center data
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # SVD
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    # principal components
    components = Vt[:k]

    # projected features
    X_reduced = X_centered @ components.T

    return X_reduced, components, X_mean

def pca_featuremap(feat, k):
    B, C, H, W = feat.shape

    # flatten spatial
    X = feat.permute(0,2,3,1).reshape(-1, C)

    X_pca, comps, mean = pca(X, k)

    # reshape back
    X_pca = X_pca.view(B, H, W, k).permute(0,3,1,2)

    return X_pca
# ----------------------------
# Paste / import your model here
# from model import preact_resnet18_bottleneck
# ----------------------------
from modules.lowrank import preact_resnet18_bottleneck
from utils.imagenet import ImageNetCustom

device = 'cpu'
model = preact_resnet18_bottleneck(num_classes=1000, in_ch=3,nblocks=2,useLR=True).to(device)
ckpt = torch.load('checkpoints_stl10_256/best_seq10.pt', map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval()
for p in model.parameters():
        p.requires_grad_(False)

train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes)
#fxtractor = create_feature_extractor(model,{"reslr.bn22":"outsconv","reslr.outlin2":"outslr"})
#fxtractor = create_feature_extractor(model,{"reslr.bn22":"outsconv","reslr.outlin2":"outslr"})
fxtractor = create_feature_extractor(model,{"reslr.bn22":"outsconv","reslr.lr2.einsum":"outslr"})
#x = torch.randn(1, 3, 256, 256)
x = torchvision.io.read_image("testimg.jpg")
x = TF.resize(x,size=288)
x = TF.center_crop(x,256)
print(x.max())
x = TF.to_tensor(x.detach().numpy())
print(x.shape)
x = x.unsqueeze(0)
x = torch.permute(x,[0,2,3,1])
features = fxtractor(x)
#feats = features["outslr"].squeeze(0)
feat_pca = pca_featuremap(features["outslr"], k=4)
feats = feat_pca.squeeze(0)
feats = feats.detach().numpy()

#meanf = features["outslr"].mean(dim=1).squeeze(0)
#plotf = torch.permute(meanf,[1,2,0])
for k in range(feats.shape[0]):
    print(f"PCA: {k}")
    plt.imshow(feats[k],cmap='jet')
    plt.show()
print(features['outslr'].shape)

#print(eval_nodes)