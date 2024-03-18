import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys
import cv2
import torch.nn.functional as F
import imageio
import math
from . import lossfunction
import time
from math import exp
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf


def gradient(input_tensor, direction):
    # print('gradient', input_tensor.shape)
    # print('gradient', input_tensor.shape)
    input_tensor = input_tensor.permute(0, 3, 1, 2)
    # print('gradient', input_tensor.shape)
    h, w = input_tensor.size()[2], input_tensor.size()[3]

    smooth_kernel_x = torch.reshape(torch.Tensor([[0., 0.], [-1., 1.]]), (1, 1, 2, 2))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)

    assert direction in ['x', 'y']
    if direction == "x":
        kernel = smooth_kernel_x
    else:
        kernel = smooth_kernel_y
    kernel = kernel.cuda()

    out = torch.nn.functional.conv2d(input_tensor, kernel, padding=(1, 1))
    out = torch.abs(out[:, :, 0:h, 0:w])
    return out.permute(0, 2, 3, 1)


def mutual_i_input_loss(input_l, input_r):  # input_l：光照  input_r：输入
    ##输入为(1,3,100,100)
    # 变为(1,100,100,3)
    # print('mutual_i_input_loss', input_l.shape)
    input_l = input_l.permute(0, 2, 3, 1)  # 变为(1,100,100,3)
    input_r = input_r.permute(0, 2, 3, 1)  # 变为(1,100,100,3)
    low_gradient_xr = gradient(input_l[:, :, :, 0:1], "x")
    input_gradient_xr = gradient(input_r[:, :, :, 0:1], "x")
    input_gradient_xr[input_gradient_xr < 0.01] = 0.01
    x_lossr = torch.abs(torch.div(low_gradient_xr, input_gradient_xr))
    low_gradient_yr = gradient(input_l[:, :, :, 0:1], "y")
    input_gradient_yr = gradient(input_r[:, :, :, 0:1], "y")
    input_gradient_yr[input_gradient_yr < 0.01] = 0.01
    y_lossr = torch.abs(torch.div(low_gradient_yr, input_gradient_yr))
    mut_lossr = torch.mean(x_lossr + y_lossr)
    return mut_lossr


def smooth_loss(self, low, I):
    dxS, dyS = tf.image.image_gradients(low)
    dxI1, dyI1 = tf.image.image_gradients(I)
    maxx = tf.maximum(tf.abs(dxS), 0.01)
    maxy = tf.maximum(tf.abs(dyS), 0.01)

    l1 = tf.div(dxI1, maxx) + tf.div(dyI1, maxy)

    loss = tf.reduce_mean(tf.abs(l1))

    return loss


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def V_c(input):
    input = input.cpu().detach().numpy()
    h = input.shape[2]
    w = input.shape[3]
    ww = np.zeros((3, h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    r = (input[0])[0]
    g = input[0][1]
    b = input[0][2]
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            V[i, j] = mx
    V = torch.tensor(V)
    ww = torch.stack((V, V, V), dim=0);
    ww = torch.tensor(ww)
    return ww


def compute_V(input, isTrian):
    if isTrian:
        x1, x2, x3, x4 = input.chunk(4, 0)
        x1 = V_c(x1)
        x2 = V_c(x2)
        x3 = V_c(x3)
        x4 = V_c(x4)
        ww = torch.stack((x1, x2, x3, x4), dim=0)
    else:
        ww = V_c(input)
        ww = torch.stack((ww, ww), dim=0)
        x1, x2 = ww.chunk(2, 0)
        return x1
    return ww


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_E = self.Tensor(nb, opt.output_nc, 600, 400)
        # self.input_V = self.Tensor(nb, opt.input_nc, size, size)
        # self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        # self.input_A_gray = self.Tensor(nb, 1, size, size)
        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
        self.d_net = networks.define_d(self.gpu_ids, skip=skip, opt=opt)
        self.h_net = networks.define_H(self.gpu_ids, skip=skip, opt=opt)
        self.s1_net = networks.define_S1(self.gpu_ids, skip=skip, opt=opt)
        # self.s2_net = networks.define_S2(self.gpu_ids, skip=skip, opt=opt)
        if not self.isTrain or opt.continue_train:
            print("---is not train----")
            which_epoch = opt.which_epoch
            print("---model is loaded---")
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.d_net, 'G_V', which_epoch)
            self.load_network(self.h_net, 'G_H', which_epoch)
            self.load_network(self.s1_net, 'S1', which_epoch)
            # self.load_network(self.s2_net, 'S2', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_d = torch.optim.Adam(self.d_net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_h = torch.optim.Adam(self.h_net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_s1 = torch.optim.Adam(self.s1_net.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_s2 = torch.optim.Adam(self.s2_net.parameters(),
            # lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.d_net)
        networks.print_network(self.h_net)
        networks.print_network(self.s1_net)
        # networks.print_network(self.s2_net)
        if opt.isTrain:
            self.netG_A.train()
            self.d_net.train()
            self.h_net.train()
            self.s1_net.train()
            # self.s2_net.train()
        else:
            self.netG_A.eval()
            self.d_net.eval()
            self.h_net.eval()
            self.s1_net.eval()
            # self.s2_net.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        input_E = input['E']
        if self.opt.isTrain:
            input_C = input['C']
        # input_V = input['V']
        # input_img = input['input_img']
        # input_A_gray = input['A_gray']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_E.resize_(input_E.size()).copy_(input_E)
        if self.opt.isTrain:
            self.input_C.resize_(input_C.size()).copy_(input_C)
        # self.input_V.resize_(input_V.size()).copy_(input_V)
        # self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def predict(self):
        nb = self.opt.batchSize
        size = self.opt.fineSize
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_E = Variable(self.input_E, volatile=True)
        flag = 0
        if self.opt.is_re:
            if self.real_A.size()[2] > 600 and self.real_A.size()[2] < 1000:
                print("donw3")
                self.down = nn.AvgPool2d(3)
                self.real_A = self.down(self.real_A)
                self.real_V = self.down(self.real_V)
                self.real_A_gray = self.down(self.real_A_gray)
                flag = 3
            elif self.real_A.size()[2] > 1000 and self.real_A.size()[2] < 2000:
                print("donw4")
                self.down = nn.AvgPool2d(4)
                self.real_A = self.down(self.real_A)
                self.real_V = self.down(self.real_V)
                self.real_A_gray = self.down(self.real_A_gray)
                flag = 4
            elif self.real_A.size()[2] > 2000 and self.real_A.size()[2] < 3000:
                print("donw5")
                self.down = nn.AvgPool2d(5)
                self.real_A = self.down(self.real_A)
                self.real_V = self.down(self.real_V)
                self.real_A_gray = self.down(self.real_A_gray)
                flag = 5
            elif self.real_A.size()[2] > 3000:
                print("donw6")
                self.down = nn.AvgPool2d(6)
                self.real_A = self.down(self.real_A)
                self.real_V = self.down(self.real_V)
                self.real_A_gray = self.down(self.real_A_gray)
                flag = 6

        # real_A = self.real_A
        # real_A.unsqueeze_(1)
        # print("self.real_A:",real_A.size())
        # real_B = self.real_B
        # real_B.unsqueeze_(1)
        # print("self.real_B:",real_B.size())
        # con = torch.cat((real_A,real_B),dim=1) 
        # print("CON:",con.size())
        self.a1 = self.d_net.forward(self.real_A)
        self.a2 = self.netG_A.forward(self.real_B)
        self.a3 = self.s1_net.forward(self.real_E)
        # print("self.a1:",self.a1.size())
        # print("self.a2:",self.a2.size())
        # print("self.real_A:",self.real_A.size())
        # print("self.real_B:",self.real_B.size())
        # real_A.squeeze_(1)
        # real_B.squeeze_(1)
        # print("self.real_A:",self.real_A.size())
        # print("self.real_B:",self.real_B.size())
        self.output1 = self.a1 * self.real_A + self.a2 * self.real_B + self.a3 * self.real_E
        self.output, self.latent = self.h_net.forward(self.output1)

        if flag == 3:
            print("up3")
            self.real_A = F.upsample(self.real_A, scale_factor=3, mode='bilinear')
            # self.fake_B = F.upsample(self.fake_B, scale_factor=3, mode='bilinear')

        elif flag == 4:
            print("up4")
            self.real_A = F.upsample(self.real_A, scale_factor=4, mode='bilinear')
            # self.fake_B = F.upsample(self.fake_B, scale_factor=4, mode='bilinear')

        elif flag == 5:
            print("up5")
            self.real_A = F.upsample(self.real_A, scale_factor=5, mode='bilinear')
            # self.fake_B = F.upsample(self.fake_B, scale_factor=5, mode='bilinear')

        elif flag == 6:
            print("up6")
            self.real_A = F.upsample(self.real_A, scale_factor=6, mode='bilinear')
            # self.fake_B = F.upsample(self.fake_B, scale_factor=6, mode='bilinear')

        U_input = util.latent2im(self.real_A.data)
        O_input = util.latent2im(self.real_B.data)
        M_input = util.latent2im(self.real_E.data)
        a1 = util.latent2im(self.a1.data)
        a2 = util.latent2im(self.a2.data)
        a3 = util.latent2im(self.a3.data)
        # a4 = util.latent2im(self.a4.data)
        # a5 = util.latent2im(self.a5.data)
        illumination = util.latent2im(self.latent.data)
        output = util.latent2im(self.output.data)
        output1 = util.latent2im(self.output1.data)
        return OrderedDict([('U_input', U_input), ('O_input', O_input), ('M_input', M_input),
                            ('A1', a1), ('A2', a2), ('A3', a3), ('illumination', illumination),
                            ('final_output', output), ('output1', output1)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)
        self.real_E = Variable(self.input_E)
        # self.real_V = Variable(self.input_V)
        # print("self.real_A:",self.real_A.size())
        # print("self.real_B:",self.real_B.size())
        # real_A = self.real_A
        # real_A.unsqueeze_(1)
        # print("self.real_A:",real_A.size())
        # real_B = self.real_B
        # real_B.unsqueeze_(1)
        # print("self.real_B:",real_B.size())
        # con = torch.cat((real_A,real_B),dim=1) 
        # print("CON:",con.size())
        self.a1 = self.d_net.forward(self.real_A)
        self.a2 = self.netG_A.forward(self.real_B)
        self.a3 = self.s1_net.forward(self.real_E)
        # print("self.a1:",self.a1.size())
        # print("self.a2:",self.a2.size())
        # print("self.real_A:",self.real_A.size())
        # print("self.real_B:",self.real_B.size())
        # real_A.squeeze_(1)
        # real_B.squeeze_(1)
        # print("self.real_A:",self.real_A.size())
        # print("self.real_B:",self.real_B.size())
        self.output1 = self.a1 * self.real_A + self.a2 * self.real_B + self.a3 * self.real_E
        self.output, self.latent = self.h_net.forward(self.output1)
        # self.a4 = self.s1_net.forward(self.real_A)
        # self.a5 = self.s2_net.forward(self.real_B)
        # self.output = self.a4*self.real_A + self.a3*self.output1 + self.a5*self.real_B

    def backward_V(self, num):
        loss_fn2 = torch.nn.MSELoss()
        # self.smooth_loss = mutual_i_input_loss(self.depth,self.real_V)
        self.V_loss_fn2 = loss_fn2(self.output1, self.real_C)
        self.V_loss_fn3 = loss_fn2(self.output, self.real_C)
        # self.V_loss_fn1 = 1-ssim(self.depth, self.real_C)
        self.loss_G_V = self.V_loss_fn2 + self.V_loss_fn3  # + 2*self.V_loss_fn1+0.2*self.smooth_loss
        self.loss_G_V.backward()

    def backward_G(self, num):
        self.G_loss_fn11 = 1 - ssim(self.di_input, self.real_B)
        self.G_loss_fn12 = 1 - ssim(self.fake_B, self.real_B)
        self.loss_G_A = self.G_loss_fn11 + self.G_loss_fn12
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                             self.di_input,
                                                             self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
            self.loss_G_A = self.loss_G_A + self.loss_vgg_b
        self.loss_G_A.backward()

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer_G.zero_grad()
        self.optimizer_d.zero_grad()
        self.optimizer_h.zero_grad()
        self.optimizer_s1.zero_grad()
        # self.optimizer_s2.zero_grad()
        self.backward_V(1)
        self.optimizer_d.step()
        self.optimizer_G.step()
        self.optimizer_h.step()
        self.optimizer_s1.step()
        # self.optimizer_s2.step()
        # self.optimizer_G.zero_grad()
        # self.backward_G(1)
        # self.optimizer_G.step()

    def get_current_errors(self, epoch):
        # G_A = self.loss_G_A.data[0]
        G_V = self.loss_G_V.data[0]
        # G_V_MSE = self.V_loss_fn2.data[0]
        # G_V_ssim = self.V_loss_fn1.data[0]
        # sl = self.smooth_loss.data[0]
        if self.opt.vgg > 0:
            return OrderedDict([('G_A', G_A), ('G_V', G_V), ('smooth_loss', sl),
                                ("G_V_MSE", G_V_MSE), ("G_V_ssim", G_V_ssim)])
        else:
            return OrderedDict([('G_V', G_V)])

    def get_current_visuals(self):
        # depth = torch.cat((self.depth,self.depth,self.depth),dim=1)
        # depth = util.latent2im(depth.data)
        # di_input =  util.latent2im(self.di_input.data)
        real_A = util.latent2im(self.real_A.data)
        real_E = util.latent2im(self.real_E.data)
        output1 = util.latent2im(self.output1.data)
        output = util.latent2im(self.output.data)
        illumination = util.latent2im(self.latent.data)
        real_B = util.latent2im(self.real_B.data)
        a1 = util.latent2im(self.a1.data)
        a2 = util.latent2im(self.a2.data)
        a3 = util.latent2im(self.a3.data)
        # a4 = util.latent2im(self.a4.data)
        # a5 = util.latent2im(self.a5.data)
        # real_C = torch.cat((self.real_C,self.real_C,self.real_C),dim=1)
        real_C = util.latent2im(self.real_C.data)
        # real_V = util.latent2im(self.real_V.data)
        # latent_real_A =  util.latent2im(self.latent_real_A.data)
        return OrderedDict([('real_A', real_A), ('real_B', real_B), ('real_C', real_C), ('real_E', real_E),
                            ('a1', a1), ('a2', a2), ('a3', a3), ('illumination', illumination),
                            ('output1', output1), ('output', output)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.d_net, 'G_V', label, self.gpu_ids)
        self.save_network(self.h_net, 'G_H', label, self.gpu_ids)
        self.save_network(self.s1_net, 'S1', label, self.gpu_ids)
        # self.save_network(self.s2_net, 'S2', label, self.gpu_ids)

    def update_learning_rate(self):

        if self.opt.new_lr:
            lr = self.old_lr / 2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_h.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_s1.param_groups:
            param_group['lr'] = lr
        # for param_group in self.optimizer_s2.param_groups:
        # param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
