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
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class upconv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, **kwargs):
        super(Encoder, self).__init__()
        self.train_mode = kwargs.get('train_mode', None)
        self.quant_bit = kwargs.get('quant_bit', 32)

        self.c1 = conv(in_dim, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c2 = conv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Sequential(
            nn.Conv2d(64, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def init_weights(self):
        generation_init_weights(self)

    def _quantize_module(self, module, bits):
        scale = 2 ** bits - 1
        for param in module.parameters():
            minimum = param.data.min()
            scaling_factor = param.data.max() - param.data.min()
            if scaling_factor > 0:
                param.data = (param.data - minimum) / scaling_factor
                param.data = torch.round(param.data * scale) / scale
                param.data = param.data * scaling_factor + minimum

    def _quantize_activation(self, x):
        scale = 2 ** self.quant_bit - 1
        minimum = x.min()
        scaling_factor = x.max() - x.min()
        if scaling_factor > 0:
            x = (x - minimum) / scaling_factor
            x = torch.round(x * scale) / scale
            x = x * scaling_factor + minimum
        return x

    def forward(self, input):
        if self.train_mode == 'qat':
            self._quantize_module(self, self.quant_bit)

        h1 = self.c1(input)
        if self.train_mode in ['qat', 'all_activations', 'encoder_activations']:
            h1 = self._quantize_activation(h1)

        h2 = self.pool1(h1)
        if self.train_mode in ['qat', 'all_activations', 'encoder_activations']:
            h2 = self._quantize_activation(h2)

        h3 = self.c2(h2)
        if self.train_mode in ['qat', 'all_activations', 'encoder_activations']:
            h3 = self._quantize_activation(h3)

        h4 = self.pool2(h3)
        if self.train_mode in ['qat', 'all_activations', 'encoder_activations']:
            h4 = self._quantize_activation(h4)

        h5 = self.c3(h4)
        if self.train_mode in ['qat', 'all_activations', 'encoder_activations']:
            h5 = self._quantize_activation(h5)
            h2 = self._quantize_activation(h2)

        return h5, h2


class Decoder(nn.Module):
    def __init__(self, out_dim=2, in_dim=32, **kwargs):
        super(Decoder, self).__init__()
        self.train_mode = kwargs.get('train_mode', None)
        self.quant_bit = kwargs.get('quant_bit', 32)

        self.conv1 = conv(in_dim, 32)
        self.upc1 = upconv(32, 16)
        self.conv2 = conv(16, 16)
        self.upc2 = upconv(32 + 16, 4)
        self.conv3 = nn.Sequential(
            nn.Conv2d(4, out_dim, 3, 1, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        generation_init_weights(self)

    def _quantize_module(self, module, bits):
        scale = 2 ** bits - 1
        for param in module.parameters():
            minimum = param.data.min()
            scaling_factor = param.data.max() - param.data.min()
            if scaling_factor > 0:
                param.data = (param.data - minimum) / scaling_factor
                param.data = torch.round(param.data * scale) / scale
                param.data = param.data * scaling_factor + minimum

    def _quantize_activation(self, x):
        scale = 2 ** self.quant_bit - 1
        minimum = x.min()
        scaling_factor = x.max() - x.min()
        if scaling_factor > 0:
            x = (x - minimum) / scaling_factor
            x = torch.round(x * scale) / scale
            x = x * scaling_factor + minimum
        return x

    def forward(self, input):
        feature, skip = input

        if self.train_mode == 'qat':
            self._quantize_module(self, self.quant_bit)

        d1 = self.conv1(feature)
        if self.train_mode in ['qat', 'all_activations', 'decoder_activations']:
            d1 = self._quantize_activation(d1)

        d2 = self.upc1(d1)
        if self.train_mode in ['qat', 'all_activations', 'decoder_activations']:
            d2 = self._quantize_activation(d2)

        d3 = self.conv2(d2)
        if self.train_mode in ['qat', 'all_activations', 'decoder_activations']:
            d3 = self._quantize_activation(d3)

        d4 = self.upc2(torch.cat([d3, skip], dim=1))
        if self.train_mode in ['qat', 'all_activations', 'decoder_activations']:
            d4 = self._quantize_activation(d4)
            skip = self._quantize_activation(skip)

        output = self.conv3(d4)
        if self.train_mode in ['qat', 'all_activations', 'decoder_activations']:
            output = self._quantize_activation(output)

        return output


class GPDL(nn.Module):
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

            quant_part = kwargs.get('quant_part', [])
            quant_bit  = kwargs.get('quant_bit', None)

            def _quantize_module(module, bits):
                scale = 2 ** bits - 1
                for param in module.parameters():
                    minimum = param.data.min()
                    scaling_factor = param.data.max() - param.data.min()
                    if scaling_factor > 0:
                        param.data = (param.data - minimum) / scaling_factor
                        param.data = torch.round(param.data * scale) / scale
                        param.data = param.data * scaling_factor + minimum

            if 'encoder' in quant_part:
                _quantize_module(self.encoder, quant_bit)
            if 'decoder' in quant_part:
                _quantize_module(self.decoder, quant_bit)
        elif pretrained is None:
            generation_init_weights(self)
        else:
            raise TypeError("'pretrained' must be a str or None. "
                            f'But received {type(pretrained)}.')
