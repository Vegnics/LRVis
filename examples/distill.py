import torch
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor

# Load pretrained model
model = resnet18(weights="IMAGENET1K_V1")
model.eval()

# Print available node names
from torchvision.models.feature_extraction import get_graph_node_names
train_nodes, eval_nodes = get_graph_node_names(model)
print(eval_nodes)

return_nodes = {
    'layer1': 'feat1',
    'layer2': 'feat2',
    'layer3': 'feat3',
    'layer4': 'feat4',
    'fc':'classvec'
}

feature_extractor = create_feature_extractor(model, ["layer1","layer2"])

x = torch.randn(1, 3, 224, 224)
features = feature_extractor(x)

for v in features.values():
    print(v.shape)