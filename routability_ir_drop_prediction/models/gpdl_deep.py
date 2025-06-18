# Copyright 2022 CircuitNet. All rights reserved.

import torch
import torch.nn as nn

from collections import OrderedDict

def generation_init_weights(module):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                    or classname.find('Linear') != -1):
            
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    module.apply(init_func)

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

class conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
        super(conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
        )

    def forward(self, input):
        return self.main(input)


class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.BatchNorm2d(dim_out),
            nn.GELU(),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=128, **kwargs):
        super(Encoder, self).__init__()

        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = conv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c4 = conv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Sequential(
            nn.Conv2d(256, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.pool1(h1) # 128x128x32
        h3 = self.c2(h2)
        h4 = self.pool2(h3) # 64x64x64
        h5 = self.c3(h4)
        h6 = self.pool3(h5) # 32x32x128
        h7 = self.c4(h6)
        h8 = self.pool4(h7) # 16x16x256
        h9 = self.c5(h8) # 16x16x128
        return h9, [h8, h6, h4, h2]


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=128, **kwargs):
        super(Decoder, self).__init__()

        self.conv1 = conv(in_dim, 128)
        self.upc1 = upconv(256 + 128, 64)
        self.conv2 = conv(64, 64)
        self.upc2  = upconv(128 + 64, 32)
        self.conv3 = conv(32, 32)
        self.upc3  = upconv(64 + 32, 16)
        self.conv4 = conv(16, 16)
        self.upc4  = upconv(32 + 16, 4)
        self.conv5 = nn.Sequential(
            nn.Conv2d(4, out_dim, 3, 1, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        generation_init_weights(self)

    def forward(self, input):
        feature, skip = input
        d1 = self.conv1(feature)
        d2 = self.upc1(torch.cat([d1, skip[0]], dim=1))
        d3 = self.conv2(d2)
        d4 = self.upc2(torch.cat([d3, skip[1]], dim=1))
        d5 = self.conv3(d4)
        d6 = self.upc3(torch.cat([d5, skip[2]], dim=1))
        d7 = self.conv4(d6)
        d8 = self.upc4(torch.cat([d7, skip[3]], dim=1))
        output = self.conv5(d8)
        return output


class GPDL_deep(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 **kwargs):
        super().__init__()
        self.encoder = Encoder(in_dim=in_channels, **kwargs)
        self.decoder = Decoder(out_dim=out_channels, **kwargs)

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

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
