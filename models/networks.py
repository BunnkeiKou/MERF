import torch
import os
import math
import torch.nn as nn

import functools
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchfile

from models.U_net import AttU_Net
from models.swinir import SwinIR

from models.gmflow.gmflow import GMFlow
import kornia


###############################################################################
# Functions
###############################################################################

def functional_conv2d(im):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_A(opt, test_mode=False):
    netA = align_FG(opt, test_mode)
    # netA.apply(weights_init)
    netA = torch.nn.DataParallel(netA)
    netA.cuda()
    return netA


def define_G(gpu_ids=[], height=320, width=320, window_size=8):
    netG = None
    use_gpu = len(gpu_ids) > 0
    # norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = SwinIR(upscale=1, img_size=(height, width), in_chans=6,
                  window_size=window_size, img_range=1., depths=[2, 2, 2],
                  embed_dim=64, num_heads=[2, 2, 2], mlp_ratio=2, upsampler='')

    # if len(gpu_ids) > 0:
    #     netG.cuda(device=gpu_ids[0])
    # 使用多个GPU加速
    netG = torch.nn.DataParallel(netG)
    netG.cuda()
    return netG


def define_Att(gpu_ids=[], skip=False, opt=None):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    net = AttING(3, 64)

    net = torch.nn.DataParallel(net)
    net.cuda()
    net.apply(weights_init)
    return net


# def define_e(gpu_ids=[], skip=False, opt=None):
#     netE = None
#     use_gpu = len(gpu_ids) > 0
#
#     if use_gpu:
#         assert (torch.cuda.is_available())
#
#     netE = eNET()
#
#     if len(gpu_ids) > 0:
#         netE.cuda(device=gpu_ids[0])
#         netE = torch.nn.DataParallel(netE, gpu_ids)
#     netE.apply(weights_init)
#     return netE


def define_GC(gpu_ids=[], skip=False, opt=None):
    netE = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netE = CEM()

    if len(gpu_ids) > 0:
        netE.cuda(device=gpu_ids[0])
        # netE = torch.nn.DataParallel(netE, gpu_ids)

    # 对于每一个模块，进行参数初始化，对于conv2d层和bn层有不同的参数初始化策略
    netE.apply(weights_init)
    return netE


def define_R(gpu_ids=[], skip=False, opt=None):
    netR = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    # netR = LF_Restore()
    netR = AttU_Net(64 * 3, 3)
    # netR = U_Net(9,3)
    netR.cuda()
    # netR = torch.nn.DataParallel(netR, gpu_ids)
    return netR

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class AttING(nn.Module):
    def __init__(self, in_channels, channels):
        super(AttING, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels, affine=True)
        self.interative = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels  # 求每个channel的标准差
        self.process = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(2 * channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        # out_instance = self.instance(x1)
        # out_identity = x1
        # # feature_save(out_identity, '1')
        # # feature_save(out_instance, '2')
        # out1 = self.conv2_1(out_instance)
        # out2 = self.conv2_2(out_identity)
        #
        # # spatial attention
        # out = torch.cat((out1, out2), 1)
        # xp1 = self.interative(out) * out2 + out1
        # xp2 = (1 - self.interative(out)) * out1 + out2
        #
        # xp = torch.cat((xp1, xp2), 1)
        # # xp = self.process(self.contrast(xp) + self.avgpool(xp)) * xp
        # xp = self.conv1x1(xp)
        # xout = xp

        return x1


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLossWGANGP():
    def __init__(self):
        self.LAMBDA = 10

    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        # DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    # def get_g_loss(self, net, realA, fakeB):
    #     # First, G(A) should fake the discriminator
    #     self.D_fake = net.forward(fakeB)
    #     return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty


# spatial attention
class BasicSpatialAttentionNet(torch.nn.Module):
    def __init__(self):
        super(BasicSpatialAttentionNet, self).__init__()

        self.fe1 = torch.nn.Conv2d(128, 64, 3, 1, 1)
        self.fe2 = torch.nn.Conv2d(64, 64, 3, 1, 1)

        self.sAtt_1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = torch.nn.Conv2d(64 * 2, 64, 1, 1, bias=True)
        self.sAtt_3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.sAtt_4 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_5 = torch.nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        self.sAtt_L1 = torch.nn.Conv2d(64, 64, 1, 1, bias=True)
        self.sAtt_L2 = torch.nn.Conv2d(64 * 2, 64, 3, 1, 1, bias=True)
        self.sAtt_L3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.htanh = Htanh()

    def forward(self, alignedframe):
        # x  =  self.fe1(alignedframe)
        # feature extraction
        att = self.lrelu(self.fe1(alignedframe))
        att = self.lrelu(self.fe2(att))

        # spatial attention
        att = self.lrelu(self.sAtt_1(att))

        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L,
                              size=[att.size(2), att.size(3)],
                              mode='bilinear', align_corners=False)  # up-sampling

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att,
                            size=[alignedframe.size(2), alignedframe.size(3)],
                            mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        # att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = F.tanh(att)
        att = att / 2 + 0.5
        # mask = self.htanh.apply(att - 0.5)
        # mask = mask / 2 + 0.5

        return att


class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, epsilon=1):
        ctx.save_for_backward(x.data, torch.tensor(epsilon))
        return x.sign()

    @staticmethod
    def backward(ctx, dy):
        x, epsilon = ctx.saved_tensors
        dx = torch.where((x < - epsilon) | (x > epsilon), torch.zeros_like(dy), dy)
        return dx, None


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image


class Gene(nn.Module):
    def __init__(self):
        super(Gene, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1)
        )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        x = self.det_conv0(input)
        x = F.relu(self.det_conv1(x))
        x = F.relu(self.det_conv2(x))
        x = F.relu(self.det_conv3(x))
        x = F.relu(self.det_conv4(x))
        return x


class Onlyillu(nn.Module):
    def __init__(self, opt, skip):
        super(Onlyillu, self).__init__()

        self.opt = opt
        self.skip = skip

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # 3
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        # self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        # 5
        self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=4, dilation=4)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)

        # 8
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=8, dilation=8)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU(inplace=True)
        # 9
        self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        # self.conv9 = nn.Conv2d(128, 128, 3, stride=1, padding=16, dilation=16)
        self.bn9 = nn.BatchNorm2d(128)
        self.relu9 = nn.ReLU(inplace=True)

        self.conv25 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn25 = nn.BatchNorm2d(128)
        self.relu25 = nn.ReLU(inplace=True)
        # 26
        self.conv26 = nn.Conv2d(128, 3, 1, stride=1, padding=0)
        self.bn26 = nn.BatchNorm2d(3)

        if self.opt.tanh:
            self.tanh = nn.Sigmoid()

    def forward(self, input):
        flag = 0

        x = self.relu1(self.bn1(self.conv1(input)))

        x = self.relu3(self.bn3(self.conv3(x)))

        res1 = x  # c3 output

        x = self.bn4(self.conv4(x))  # r4
        x = self.relu4(x)

        x = self.bn5(self.conv5(x))  # fr5
        x = self.relu5(x + res1)  # tr5
        res3 = x

        x = self.bn8(self.conv8(x))
        x = self.relu8(x)

        x = self.bn9(self.conv9(x))
        x = self.relu9(x)
        res7 = x

        # x = self.bn20(self.conv20(x))
        # x = self.relu20(x + res7)
        # res18 = x

        # x = self.bn21(self.conv21(x))
        # x = self.relu21(x + res18)
        # res19 = x

        # x = self.relu24(self.bn24(self.deconv24(x)))
        # x = self.relu25(self.bn25(self.conv25(x)))
        x = self.bn25(self.conv25(x))
        x = self.relu9(x + res7)
        latent = self.conv26(x)

        if self.opt.tanh:
            latent = self.tanh(latent)
        output = input / (latent + 0.00001)
        return latent, output


class LF_Restore(nn.Module):
    def __init__(self, channels=9):
        super(LF_Restore, self).__init__()
        # self.d1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, dilation=1)
        # self.d2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, dilation=2, padding=2)
        # self.sigmoid = nn.Sigmoid()

        self.self_att = AttU_Net(9, 3)

    def forward(self, x):
        x1 = x[:, 6:, :, :]
        # d1 = self.d1(x)
        # d2 = self.d2(x)
        # Ca = self.sigmoid(d1-d2)
        # x_lf = (1-Ca)*x
        # att = self.self_att(x_lf)
        att = self.self_att(x)
        result = att * x1
        return result


class DnCNN(nn.Module):
    def __init__(self, channels=9, num_of_layers=15):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.LeakyReLU(negative_slope=0.8, inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.LeakyReLU(negative_slope=0.8, inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=3, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class CEM(nn.Module):
    def __init__(self):
        super(CEM, self).__init__()
        self.inc = DoubleConV(9, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 64)
        # self.down3 = Down(128, 256)
        # factor = 2
        # factor = 1
        # self.down3 = Down(128, 256 // factor)
        # self.down4 = Down(256, 512 // factor)
        # self.up1 = Up(512, 256 // factor)
        self.up1 = Up(128, 32)
        self.up2 = Up(64, 32)
        # self.up3 = Up(64, 32)
        self.outc = OutConV(32, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.up3(x, x1)
        # x = self.up4(x, x1)
        out = self.outc(x)
        out = F.tanh(out)
        out = out / 2 + 0.5

        return out


class DoubleConV(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pool='Max'):
        super().__init__()
        if pool == 'Max':
            self.pool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConV(in_channels, out_channels)
            )
        else:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                DoubleConV(in_channels, out_channels)
            )

    def forward(self, x):
        return self.pool_conv(x)


class Up_Y(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConV(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = x2 + x1
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConV(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat((x2, x1), dim=1)
        return self.conv(x)


class OutConV(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConV, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DEM(nn.Module):
    def __init__(self):
        super(DEM, self).__init__()
        self.inc_l = DoubleConV(3, 16)
        self.down1_l = Down(16, 32)
        self.down2_l = Down(32, 64)

        self.inc_h = DoubleConV(3, 16)
        self.down1_h = Down(16, 32)
        self.down2_h = Down(32, 64)

        self.inc_2 = DoubleConV(6, 32)
        self.down1_2 = Down(32, 64, 'Avg')
        self.down2_2 = Down(64, 128, 'Avg')

        # factor = 1
        self.DoubleConv = DoubleConV(256, 128, 256 // 2)

        self.up1 = Up(256, 64)
        self.up2 = Up(128, 32)

        self.outc = OutConV(32, 1)

    def forward(self, input):
        img1 = input[:, 0:3, :, :]  # low
        img2 = input[:, 3:, :, :]  # high

        x1_1 = self.inc_l(img1)  # 16
        x2_1 = self.down1_l(x1_1)  # 32
        x3_1 = self.down2_l(x2_1)  # 64

        x1_2 = self.inc_h(img2)
        x2_2 = self.down1_h(x1_2)
        x3_2 = self.down2_h(x2_2)

        j1 = self.inc_2(input)
        j2 = self.down1_2(j1)
        j3 = self.down2_2(j2)  # 128

        x = self.DoubleConv(torch.cat([x3_1, x3_2, j3], dim=1))  # 256

        x = self.up1(x, torch.cat([x2_1, x2_2, j2], dim=1))
        x = self.up2(x, torch.cat([x1_1, x1_2, j1], dim=1))

        out = self.outc(x)

        out = F.tanh(out)
        out = out / 2 + 0.5

        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    # haodeh assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    grid = kornia.utils.create_meshgrid(h, w, device=x.device).to(x.dtype)
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4':  # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        output = F.grid_sample(x, vgrid, mode=interp_mode, padding_mode=padding_mode,
                               align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
    return output


def flow_warp2(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.
        use_pad_mask (bool): only used for PWCNet, x is first padded with ones along the channel dimension.
            The mask is generated according to the grid_sample results of the padded dimension.


    Returns:
        Tensor: Warped image or feature map.
    """
    # haodeh assert x.size()[-2:] == flow.size()[1:3] # temporaily turned off for image-wise shift
    n, _, h, w = x.size()
    x = x.float()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x)) # an illegal memory access on TITAN RTX + PyTorch1.9.1
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow

    # if use_pad_mask: # for PWCNet
    #     x = F.pad(x, (0,0,0,0,0,1), mode='constant', value=1)

    # scale grid to [-1,1]
    if interp_mode == 'nearest4':  # todo: bug, no gradient for flow model in this case!!! but the result is good
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)

        return torch.cat([output00, output01, output10, output11], 1)

    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode,
                               align_corners=align_corners)

        # if use_pad_mask: # for PWCNet
        #     output = _flow_warp_masking(output)

        # TODO, what if align_corners=False
        return output


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x1 = self.conv(x)
        out = torch.sigmoid(x1)
        return out


class FuseModule(nn.Module):
    """ Interactive fusion module"""

    def __init__(self, in_dim=64):
        super(FuseModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, prior):
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


class align_FG(nn.Module):
    def __init__(self, opt, test_mode=False):
        super(align_FG, self).__init__()
        # self.F_ref = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        # self.F_move = nn.Conv2d(3, 3, 3, padding=1, bias=True)
        # self.FeatureEx = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        # self.FeatureExR = nn.Conv2d(3, 64, 3, padding=1, bias=True)
        # self.mask = nn.Conv2d(64, 2, 3, padding=1, bias=True)
        # self.att = nn.Sequential(
        #     nn.Conv2d(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
        #     nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1),
        #     nn.Sigmoid()
        # )
        # self.align = PCD_Align()
        # self.align = FlowGuidedPCDAlign()
        # self.recon = nn.Conv2d(64, 3, 1, 1, 0)
        # deformable
        self.relu = nn.ReLU(True)
        # self.ref_fuse = nn.Conv2d(128, 64, 3, 1, 1)
        # self.select = FuseModule()
        self.test_mode = test_mode
        if test_mode:
            self.flow_net = GMFlow(feature_channels=128,
                                   num_scales=2,
                                   upsample_factor=4,
                                   num_head=1,
                                   attention_type='swin',
                                   ffn_dim_expansion=4,
                                   num_transformer_layers=6,
                                   )
        else:
            self.flow_net = GMFlow(feature_channels=128,
                                   num_scales=2,
                                   upsample_factor=4,
                                   num_head=1,
                                   attention_type='swin',
                                   ffn_dim_expansion=4,
                                   num_transformer_layers=6,
                                   )
        # self.occ_check_model = tools.occ_check_model(occ_type='for_back_check', occ_alpha_1=0.1,
        #                                              occ_alpha_2=0.5,
        #                                              obj_out_all='obj')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        # # xavier initialization

        # self.RAFT = RAFT(opt)
        # self.RAFT.cuda()
        # self.RAFT.eval()
        # new_state_dict = {}
        # checkpoint = torch.load('models/raft-things.pth')
        # for k, v in checkpoint.items():
        #     new_state_dict[k[7:]] = v
        # self.RAFT.load_state_dict(new_state_dict)
        # # self.RAFT.load_state_dict(torch.load('models/raft-things.pth'))
        # # self.fw = forward_warp()
        # for param in self.RAFT.parameters():
        #     param.requires_grad = False
        # self.patch_embed = PatchEmbed(patch_size=1)
        # self.patch_unembed = PatchUnEmbed(patch_size=1,embed_dim=64)
        # depths = [2]
        # drop_path_rate = 0.1
        # patches_resolution = self.patch_embed.patches_resolution  # 分割得到patch的分辨率
        # self.patches_resolution = patches_resolution
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # self.select_feature = CRSTB(dim=64,
        #                             input_resolution=(patches_resolution[0],
        #                                               patches_resolution[1]),
        #                             depth=2,
        #                             num_heads=2,
        #                             window_size=8,
        #                             mlp_ratio=2,
        #                             drop_path=dpr[sum(depths[:0]):sum(depths[:1])],
        #                             # no impact on SR results
        #                             downsample=None,
        #                             img_size=64,
        #                             patch_size=1,
        #                            )
        # drop_rate = 0.
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.norm = nn.LayerNorm(64)

    def forward(self, x, ref, x_ins):
        # ref_cor = self.relu(self.F_ref(ref)) + ref
        # x = self.F_move(x)
        # high = self.pyramid_feats(high)
        # x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        # x_norm = F.normalize((x - x_mean))
        # ref_mean = torch.mean(ref, dim=[2, 3], keepdim=True)
        # ref_norm = F.normalize((ref - ref_mean))
        # down_sample = nn.AvgPool2d
        # r_x = self.recon(x)
        # r_ref = self.recon(ref)
        # init_flow, flow_R = self.RAFT(r_ref, r_x, iters=20, test_mode=True)
        # # forward
        # x_warp = flow_warp(x, flow_R.permute(0, 2, 3, 1))

        ref_x = self.flow_net(ref, x, attn_splits_list=[2, 8],
                              corr_radius_list=[-1, 4],
                              prop_radius_list=[-1, 4],
                              )
        f = ref_x['flow_preds']
        F_warped_move_ = flow_warp2(x_ins, f[-1].permute(0, 2, 3, 1))
        # x_out, ref_out = self.select(F_warped_move_, ref_ins)

        x_ref = self.flow_net(x, ref, attn_splits_list=[2, 8],
                              corr_radius_list=[-1, 4],
                              prop_radius_list=[-1, 4],
                              )
        f_sup = x_ref['flow_preds']
        return F_warped_move_, f, f_sup


def define_F(gpu_ids=[], skip=False, opt=None):

    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    net = FuseModule()

    net = torch.nn.DataParallel(net)
    net.cuda()
    net.apply(weights_init)
    return net

