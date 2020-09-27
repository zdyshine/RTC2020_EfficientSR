import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class GCT(nn.Module):
    '''
    论文链接: https://arxiv.org/abs/1909.11519
    代码地址: https://github.com/z-x-yang/GCT
    '''
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''

        return pixel_unshuffle(input, self.downscale_factor)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class IMDModule(nn.Module):
    '''
    FEDB: Feature Enhance Distill Block
    '''
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate) # 12 * 0.25 = 3
        self.remaining_channels = int(in_channels - self.distilled_channels) # 9
        self.c1 = conv_layer(in_channels, 48, 3) # 12 -> 48
        self.c2 = conv_layer(45, 33, 3) # 45 -> 33
        self.c3 = conv_layer(30, 12, 3) # 30 -> 12
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(30, in_channels, 1) # 30 -> 12
        self.gct = GCT(num_channels=30)

    def forward(self, input):
        out_c1 = self.act(self.c1(input)) # 12 -> 48
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, 45), dim=1) # 3, 45
        out_c2 = self.act(self.c2(remaining_c1)) #  45 -> 33
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, 30), dim=1) # 3, 30
        out_c3 = self.act(self.c3(remaining_c2)) # 30 -> 12

        out = torch.cat([distilled_c1, distilled_c2, out_c3, input], dim=1)
        out_fused = self.c5(self.gct(out))
        # out_fused = self.c5(out)
        return out_fused

class IMDModule_So(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_So, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = conv_layer(in_channels, 27, 3)
        self.c2 = conv_layer(24, 18, 3)
        self.c3 = conv_layer(15, 9, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(30, in_channels, 1)

    def forward(self, input, x_sobel):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, 24), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, 15), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))

        out = torch.cat([distilled_c1, distilled_c2, out_c3, input, x_sobel], dim=1)
        out_fused = self.c5(out)
        return out_fused

class GhostModule(nn.Module):
    '''
    论文链接: https://arxiv.org/abs/1911.11907
    代码地址: https://github.com/huawei-noah/ghostnet
    '''
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

class SRB(nn.Module):
    def __init__(self, nf):
        super(SRB, self).__init__()
        self.c = conv_layer(nf, nf, 3)
        self.act = activation('lrelu', neg_slope=0.05)
    def forward(self, x):
        return self.act(x + self.c(x))


class SESF(nn.Module):
    '''
    AIM 2020 Efficient SR
    '''
    def __init__(self, nf=12, squeeze=9, expand=128):
        super(SESF, self).__init__()
        assert expand - squeeze > 12, 'channel error expand - squeeze should > 12 '
        self.remaining_channels = expand - nf * 2 # 104
        self.distilled_channels = nf * 2 # 24
        self.c1 = conv_layer(nf, expand, 1) # 12 ->128
        self.c2_1 = conv_layer(self.remaining_channels, squeeze, 1) # 128-24=104 ->9
        self.c2_2 = conv_layer(self.distilled_channels, nf, 3) # 24 -> 12

        self.c3 = conv_layer(nf * 3, nf, 1)
        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x, x_xobel):
        out_c1 = self.act(self.c1(x)) # 12 -> 128
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1) # 24, 104
        out_c1 = self.c2_1(remaining_c1) # 9 out
        out_c1_1 = self.c2_2(distilled_c1)# 12

        out = torch.cat([x, out_c1, out_c1_1, x_xobel], dim=1)
        out_fused = self.act(self.c3(out))
        return out_fused

class model_rtc(nn.Module):
    def __init__(self, upscale=2, in_nc=3, nf=12, num_modules=4, out_nc=3):
        super(model_rtc, self).__init__()

        self.fea_conv = conv_layer(nf, nf, kernel_size=3)
        self.gct_in = GCT(num_channels=nf)

        self.rb_blocks1 = IMDModule(in_channels=nf)
        self.rb_blocks2 = IMDModule(in_channels=nf)
        self.rb_blocks6 = IMDModule_So(in_channels=nf)

        self.gct = GCT(num_channels=nf*3)
        self.shirking = conv_layer(nf*3, nf, kernel_size=1)

        upsample_block = pixelshuffle_block
        self.upsampler1 = upsample_block(nf, nf, upscale_factor=upscale)
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)

    def forward(self, input):
        '''
        conv + gct(cat后加)
        Flops:  1.96 GMac
        Params: 61.53 k
        '''
        x = input
        input = pixel_unshuffle(x, 2)
        input = self.fea_conv(input)
        x1 = self.rb_blocks1(self.gct_in(input))
        x2 = self.rb_blocks2(x1)
        out = torch.cat([input, x1, x2], dim=1)
        out = self.shirking(self.gct(out))
        out = self.upsampler1(out)

        out = self.rb_blocks6(out, x)
        out = self.upsampler(out)
        input = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return input + out
