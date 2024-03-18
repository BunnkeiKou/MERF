import torch
from torch import nn

import util.util as util
from collections import OrderedDict
from torch.autograd import Variable

from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from math import exp
import kornia


def fuse_Cx(A, B):
    A = A * 255
    B = B * 255
    A_128 = abs(A - 128)
    B_128 = abs(B - 128)
    fuse = (A * A_128 + B * B_128) / (A_128 + B_128)
    fuse[torch.isnan(fuse)] = 128
    return fuse / 255.0


def sequence_loss(flow_preds, flow_gt, gamma=0.9, max_flow=400):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    # valid = (valid >= 0.5) & (mag < max_flow)
    valid = mag < max_flow

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    return flow_loss


# Laplacian算子，图像增强，边缘提取
def functional_conv2d(im):
    sobel_kernel = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    sobel_kernel = torch.reshape(sobel_kernel, ((1, 1, 3, 3)))
    sobel_kernel = sobel_kernel.cuda()  # 加载到GPU上运算
    weight = Variable(sobel_kernel)
    edge_detect = F.conv2d(Variable(im), weight, stride=1, padding=1)
    # edge_detect = edge_detect.squeeze().detach().cpu().numpy()
    return edge_detect


def gradient_im(im):
    sobel_kernel = torch.Tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    sobel_kernel = torch.reshape(sobel_kernel, ((1, 1, 3, 3)))
    sobel_kernel = sobel_kernel.cuda()  # 加载到GPU上运算
    weight = sobel_kernel
    gradient_im = F.conv2d(im, weight, stride=1, padding=1)
    # edge_detect = edge_detect.squeeze().detach().cpu().numpy()
    return gradient_im


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def gaussian_2D(img, window_size=3, padding=1):
    (_, channel, height, width) = img.size()
    real_size = min(window_size, height, width)
    window = create_window(real_size, channel=channel).to(img.device)
    pad = nn.ReflectionPad2d(padding)
    img = pad(img)
    result = F.conv2d(img, window, groups=channel)

    return result


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


def flow_warp2(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True, use_pad_mask=False):
    n, _, h, w = x.size()
    x = x.float()

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow

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

        # TODO, what if align_corners=False
        return output


class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    # opt为参数类
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        # self.input_A2 = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_B2 = self.Tensor(nb, opt.output_nc, 600, 400)
        self.flow = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C = self.Tensor(nb, opt.output_nc, 600, 400)

        if opt.vgg > 0:
            # 它是将真实图片卷积得到的feature（一般是用vgg16或者vgg19来提取）与生成图片卷积得到的feature作比较（一般用MSE损失函数），使得高层信息（内容和全局结构）接近，也就是感知的意思。
            # 在超分中，因为我们经常使用MSE损失函数，会导致输出图片比较平滑（丢掉了细节部分/高频部分），因此适当选择某个层输出的特征输入感知损失函数是可以增强细节
            self.vgg_loss = networks.PerceptualLoss(opt)  # 感知损失
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            # 类型转换
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model", self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        self.avg_pool = nn.AvgPool2d(2, 2)

        self.netA = networks.define_A(opt)
        self.netA.load_state_dict(torch.load("./checkpoints/Final_Flow/800_net_A.pth"), strict=True)
        window_size = 4
        self.Mapping = networks.define_Att()
        self.Dual_Att = networks.define_F()

        self.Mapping.load_state_dict(torch.load("./checkpoints/Final_Flow/800_net_M.pth"), strict=True)

        self.netG = networks.define_G(gpu_ids=[], window_size=window_size)
        self.netG.load_state_dict(torch.load("./checkpoints/Ex_S_SICE/400_net_G.pth"), strict=True)

        self.refinement_net = networks.define_R()
        self.refinement_net.load_state_dict(torch.load("./checkpoints/Ex_S_SICE/400_net_R.pth"), strict=True)


        if not self.isTrain or opt.continue_train:
            print("---is not train----")
            which_epoch = opt.which_epoch
            print("---model is loaded---")
            self.load_network(self.netG, 'G', which_epoch)
            self.load_network(self.Dual_Att, 'D', which_epoch)
            self.load_network(self.Mapping, 'M', which_epoch)
            self.load_network(self.netA, 'A', which_epoch)
            self.load_network(self.refinement_net, 'R', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr

            self.optimizer_M = torch.optim.Adam(self.Mapping.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.Dual_Att.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_R = torch.optim.Adam(self.refinement_net.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')

        if opt.isTrain:
            self.netG.train()
            self.Mapping.train()
            self.netA.train()
            self.Dual_Att.train()
            self.refinement_net.train()
        else:
            self.netG.eval()
            self.Mapping.eval()
            self.netA.eval()
            self.Dual_Att.eval()
            self.refinement_net.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'


        self.input_A.resize_(input['A'].size()).copy_(input['A'])


        self.input_B.resize_(input['B'].size()).copy_(input['B'])
        self.input_B2.resize_(input['B2'].size()).copy_(input['B2'])
        self.flow.resize_(input['flow'].size()).copy_(input['flow'])


        if self.opt.isTrain:
            # input_C = YCbCr_transformer(input['C'])
            self.input_C.resize_(input['C'].size()).copy_(input['C'])

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def forward(self):

        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.identity_A, self.correct_A = self.Mapping.forward(self.input_A)

        self.identity_B, self.correct_B = self.Mapping.forward(self.input_B)

        self.align_B, self.f_w, self.f_s = self.netA.forward(self.correct_B, self.correct_A, self.identity_B)

        self.align_B, self.align_A = self.Dual_Att.forward(self.align_B, self.identity_A)
        self.img_level = flow_warp2(self.input_B, self.f_w[-1].permute(0, 2, 3, 1))

        self.align_B_half = self.avg_pool(self.align_B)
        self.align_A_half = self.avg_pool(self.align_A)

        self.o_lf, self.x_lf = self.netG.forward(self.align_B_half, self.align_A_half)
        self.o_lf = up(self.o_lf)
        self.x_lf = up(self.x_lf)

        self.detail = self.refinement_net.forward(self.align_B, self.align_A, self.x_lf)

        self.refinement = self.detail + self.o_lf


