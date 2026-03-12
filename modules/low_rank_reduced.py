


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(pretrained=True)

        self.layer1 = copy.deepcopy(resnet.layer1)
        self.layer2 = copy.deepcopy(resnet.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x