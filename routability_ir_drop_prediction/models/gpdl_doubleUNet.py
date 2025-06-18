import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
import pdb
"""
This file contains an **end-to-end implementation of Double-UNet** where the
first encoder is a **frozen ResNet-50** (pre-trained on ImageNet) instead of the
original VGG-19.  All other ideas from the Double-UNet paper are preserved:

* ASPP after each encoder's bottleneck
* squeeze-and-excite (SE) blocks inside every conv block
* element-wise multiplication of Network-1's prediction with the input image
  before feeding Network-2
* skip-connections from Encoder-1 to Decoder-2 in addition to those coming
  from Encoder-2 (dashed arrows in the original figure).

The code is intentionally **modular** so you can swap blocks easily.
"""

class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.se = SELayer(out_channels)

    def forward(self, x):
        return self.se(self.main(x))


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.up(x)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, rates=(6, 12, 18)):
        super().__init__()
        
        # 1×1 convolution branch (no dilation)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Three 3×3 branches with different dilation rates
        self.branches = nn.ModuleList()
        for r in rates:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=r,
                        dilation=r,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()
                )
            )

        # Image‐level pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Projection: reduce (N_branches + 1) * out_channels → out_channels
        num_branches = 1 + len(rates)  # 1×1 conv branch + len(rates) dilated branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (num_branches + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 1×1 branch
        out_1x1 = self.branch1(x)  # → (B, out_channels, H, W)

        # Dilated branches
        out_dilated = []
        for branch in self.branches:
            out_dilated.append(branch(x))  # each → (B, out_channels, H, W)

        # Image-level pooling
        pooled = self.global_pool(x)        # → (B, out_channels, 1, 1)
        pooled = F.interpolate(
            pooled,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )                                   # → (B, out_channels, H, W)

        # Concatenate: [1×1, dilated@r=6, ... , dilated@r=18, pooled]
        all_feats = [out_1x1] + out_dilated + [pooled]
        x_cat = torch.cat(all_feats, dim=1) # → (B, out_channels *  (num_branches + 1), H, W)

        # Project back to out_channels
        return self.project(x_cat)          # → (B, out_channels, H, W)

class ResNet50Encoder(nn.Module):
    """Frozen ImageNet ResNet-50 that returns 4 skip maps + bottleneck."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)


        for p in resnet.parameters():
            p.requires_grad = False

        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        c0 = self.initial(x) # 128x128x64
        c1 = self.layer1(self.maxpool(c0)) # 64x64x256
        c2 = self.layer2(c1) # 32x32x512
        c3 = self.layer3(c2) # 16x16x1024
        c4 = self.layer4(c3) # 8x8x2048
        return c4, [c3, c2, c1, c0]

class VanillaEncoder(nn.Module):
    """A plain encoder used for the 2nd UNet."""

    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        self.c1 = ConvBlock(in_channels, base_ch)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = ConvBlock(base_ch, base_ch * 4)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = ConvBlock(base_ch * 4, base_ch * 8)
        self.p3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c4 = ConvBlock(base_ch * 8, base_ch * 16)
        self.p4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = ConvBlock(base_ch * 16, base_ch * 32)

    def forward(self, x):
        s1 = self.c1(x) # 128x128x64
        s2 = self.c2(self.p1(s1)) # 64x64x256
        s3 = self.c3(self.p2(s2)) # 32x32x512
        s4 = self.c4(self.p3(s3)) # 16x16x1024
        bottleneck = self.c5(self.p4(s4)) # 8x8x2048
        return bottleneck, [s4, s3, s2, s1]


class Decoder(nn.Module):
    """UNet-style 4-stage decoder.

    *bottleneck_ch* - channel count of the input feature map.
    *skip_chs*      - list with 4 ints (deep → shallow) describing the number
                      of channels in **each** skip connection that will be
                      provided **after concatenation** (if multiple encoders).
    *out_channels*        - number of output channels.
    """

    def __init__(self, bottleneck_ch: int, skip_chs: list[int], out_channels: int):
        super().__init__()
        assert len(skip_chs) == 4, "skip_chs must list 4 elements (deepest→shallow)."

        self.up4 = UpConv(bottleneck_ch, 1024)
        self.dec4 = ConvBlock(1024 + skip_chs[0], 1024)

        self.up3 = UpConv(1024, 512)
        self.dec3 = ConvBlock(512 + skip_chs[1], 512)

        self.up2 = UpConv(512, 256)
        self.dec2 = ConvBlock(256 + skip_chs[2], 256)

        self.up1 = UpConv(256, 128)
        self.dec1 = ConvBlock(128 + skip_chs[3], 64)

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, bottleneck, skips):
        # compress skip channels first
        d4 = self.dec4(torch.cat([self.up4(bottleneck), skips[0]], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skips[1]], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skips[2]], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skips[3]], dim=1))
        return self.out_conv(d1)

#####################################################################
#                        Double-UNet wrapper                         #
#####################################################################

class GPDL_doubleUNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, **kwargs):
        super().__init__()

        # ------------------- Network-1 -------------------
        self.enc1 = ResNet50Encoder()
        self.aspp1 = ASPP(2048, 1024)
        self.dec1 = Decoder(bottleneck_ch=1024,
                             skip_chs=[1024, 512, 256, 64],
                             out_channels=3)

        # ------------------- Network-2 -------------------
        self.input_transform = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.enc2 = VanillaEncoder(in_channels=in_channels, base_ch=64)
        self.aspp2 = ASPP(2048, 1024)

        # skip channels for Decoder-2 comes from **both** encoders ⇒ concatenate.
        self.dec2 = Decoder(bottleneck_ch=1024,  # from aspp2
                             skip_chs=[2048, 1024, 512, 128],  # enc2 + enc1
                             out_channels=3)
        
        self.output_transform = UpConv(in_channels=6, out_channels=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return **two full-resolution masks** stacked along the channel axis."""
        # ---------- Network-1 ----------
        bottleneck1, skips1 = self.enc1(x)
        output1 = self.dec1(self.aspp1(bottleneck1), skips1)

        # ---------- Network-2 ----------
        x2 = self.input_transform(x) * output1
        bottleneck2, skips2 = self.enc2(x2)
        # merge skips:   enc2 (high-level) ⨁ enc1 (compressed) - must match spatial sizes
        merged_skips = [
            torch.cat([sk2, sk1], dim=1)  # channel-wise cat, spatial dims already aligned
            for sk2, sk1 in zip(skips2, skips1)
        ]

        output2 = self.dec2(self.aspp2(bottleneck2), merged_skips)          # H × W  (already full-res)

        # ----------- return both heads -----------
        return self.output_transform(torch.cat([output1, output2], dim=1))  # (N, 4, H, W) [mask1, mask2], dim=1)  # (B, in_channels+out_channels, H, W)
    
    def init_weights(self, pretrained=None, pretrained_transfer=None, strict=False, **kwargs):
        if isinstance(pretrained, str):
            new_dict = OrderedDict()
            weight = torch.load(pretrained, map_location='cpu')['state_dict']
            for k in weight.keys():
                new_dict[k] = weight[k]
            load_state_dict(self, new_dict, strict=strict, logger=None)

        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')

#####################################################################
#                   optional weight initialisation                  #
#####################################################################

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    module.apply(init_func)


def freeze_bn(module):
    """Freeze BatchNorm layers - useful when fine-tuning with small batches."""
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


def load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None

    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    return missing_keys

