import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import Permute
from torchvision.ops.stochastic_depth import StochasticDepth
import torchinfo


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_gelu=False):
        super().__init__()
        self.use_gelu = use_gelu
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])
        if self.use_gelu:
            self.gelu = nn.GELU()

    def forward(self, x):
        x = self.permute1(x)
        x = self.linear(x)
        if self.use_gelu:
            x = self.gelu(x)
        x = self.permute2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, features, ):
        super().__init__()
        self.norm = nn.LayerNorm(features)
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.norm(x)
        x = self.permute2(x)
        return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()
        self.permute1 = Permute([0, 2, 3, 1])
        self.permute2 = Permute([0, 3, 1, 2])

    def forward(self, x):
        x = self.permute1(x)
        x = self.gelu(x)
        x = self.permute2(x)
        return x


class CNBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, padding=3, dilation=1, layer_scale=1e-6, stochastic_depth_prob=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class DAPPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_block1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                        Linear(in_channels, out_channels))
        self.avg_block2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                        Linear(in_channels, out_channels))
        self.avg_block3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                        Linear(in_channels, out_channels))
        self.avg_block4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Linear(in_channels, out_channels))

        self.conv0 = Linear(in_channels, out_channels)
        self.conv1 = nn.Sequential(LayerNormalization(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = nn.Sequential(LayerNormalization(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv3 = nn.Sequential(LayerNormalization(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv4 = nn.Sequential(LayerNormalization(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv5 = nn.Sequential(LayerNormalization(out_channels),
                                   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self.cat = Linear(out_channels * 5, out_channels)
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x):
        height, width = x.shape[-2], x.shape[-1]
        x0 = self.conv0(x)

        x1 = self.avg_block1(x)
        x1 = F.interpolate(x1, size=[height, width], mode='bilinear')
        x1 = self.conv1(x1 + x0)

        x2 = self.avg_block2(x)
        x2 = F.interpolate(x2, size=[height, width], mode='bilinear')
        x2 = self.conv2(x2 + x1)

        x3 = self.avg_block3(x)
        x3 = F.interpolate(x3, size=[height, width], mode='bilinear')
        x3 = self.conv3(x3 + x2)

        x4 = self.avg_block4(x)
        x4 = F.interpolate(x4, size=[height, width], mode='bilinear')
        x4 = self.conv4(x4 + x3)

        x_ = self.cat(torch.cat([x0, x1, x2, x3, x4], dim=1))
        return x_ + self.linear(x)


class Model(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).features

        self.encoder = IntermediateLayerGetter(self.encoder, return_layers={"1": "feature4",
                                                                            "3": "feature8",
                                                                            "5": "feature16",
                                                                            "7": "feature32"})
        self.conv4 = nn.Sequential(LayerNormalization(768), nn.Conv2d(768, 96, kernel_size=3, stride=1, padding=1),
                                   GELU())
        self.conv3 = nn.Sequential(LayerNormalization(384), nn.Conv2d(384, 96, kernel_size=3, stride=1, padding=1),
                                   GELU())
        self.conv2 = nn.Sequential(LayerNormalization(192), nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1),
                                   GELU())
        self.conv1 = nn.Sequential(LayerNormalization(96), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                   GELU())

        self.conv4up = nn.Sequential(LayerNormalization(96), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                     GELU(), nn.Upsample(scale_factor=8, mode="bilinear"))
        self.conv3up = nn.Sequential(LayerNormalization(96), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                     GELU(), nn.Upsample(scale_factor=4, mode="bilinear"))
        self.conv2up = nn.Sequential(LayerNormalization(96), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                     GELU(), nn.Upsample(scale_factor=2, mode="bilinear"))
        self.conv1up = nn.Sequential(LayerNormalization(96), nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
                                     GELU())


        self.aux_head = nn.Sequential(Linear(96, num_classes),
                                      nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False))
        self.out_head = nn.Sequential(Linear(96, num_classes),
                                      nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False))

        self.cat = nn.Sequential(LayerNormalization(96 * 4), nn.Conv2d(96 * 4, 96, kernel_size=3, stride=1, padding=1),
                                 GELU(), Linear(96, 96))

        self.dappm = DAPPM(768, 96)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, image):
        outputs = OrderedDict()
        features = self.encoder(image)
        x4 = self.conv4(features["feature32"]) + self.dappm(features["feature32"])
        x3 = self.conv3(features["feature16"]) + self.up(x4)
        x2 = self.conv2(features["feature8"]) + self.up(x3)
        x1 = self.conv1(features["feature4"]) + self.up(x2)

        outputs["aux"] = self.aux_head(x1)

        x4 = self.conv4up(x4)
        x3 = self.conv3up(x3)
        x2 = self.conv2up(x2)
        x1 = self.conv1up(x1)

        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.cat(x)
        outputs["out"] = self.out_head(x)

        return outputs


if __name__ == "__main__":
    img = torch.randn(8, 3, 256, 256)
    un = DeepLabNext(4)
    out = un(img)
    print(count_parameters(un))
    print([(k, v.shape) for k, v in out.items()])
    print(torchinfo.summary(un, (64, 3, 256, 256), device="cpu",
                            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
                            row_settings=["var_names"], ))
