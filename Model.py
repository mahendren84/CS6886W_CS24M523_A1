import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":    return nn.ReLU(inplace=True)
    if name == "silu":    return nn.SiLU(inplace=True)
    if name == "gelu":    return nn.GELU()
    if name == "tanh":    return nn.Tanh()
    if name == "sigmoid": return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, act: nn.Module, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 3, padding=1, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(out_c))
        layers.append(act)
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class VGG6(nn.Module):
    def __init__(self, num_classes=10, activation="relu", dropout=0.3, batch_norm=True):
        super().__init__()
        A = lambda: get_activation(activation)
        self.features = nn.Sequential(
            ConvBlock(3,   64, A(), batch_norm), ConvBlock(64,  64, A(), batch_norm), nn.MaxPool2d(2),
            ConvBlock(64, 128, A(), batch_norm), ConvBlock(128,128, A(), batch_norm), nn.MaxPool2d(2),
            ConvBlock(128,256, A(), batch_norm), ConvBlock(256,256, A(), batch_norm), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), get_activation(activation), nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))
