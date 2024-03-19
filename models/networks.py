import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from models.U_net import AttU_Net
from models.swinir import SwinIR
from models.gmflow.gmflow import GMFlow
import kornia


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_A(opt, test_mode=False):
    netA = align_FG(opt, test_mode)
    netA = torch.nn.DataParallel(netA)
    netA.cuda()
    return netA


def define_G(gpu_ids=[], height=320, width=320, window_size=8):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = SwinIR(upscale=1, img_size=(height, width), in_chans=6,
                  window_size=window_size, img_range=1., depths=[2, 2, 2],
                  embed_dim=64, num_heads=[2, 2, 2], mlp_ratio=2, upsampler='')

    netG = torch.nn.DataParallel(netG)
    netG.cuda()
    return netG


def define_Att(gpu_ids=[]):
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    net = AttING(3, 64)

    net = torch.nn.DataParallel(net)
    net.cuda()
    net.apply(weights_init)
    return net


def define_R(gpu_ids=[]):
    netR = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())

    netR = AttU_Net(64 * 3, 3)
    netR.cuda()

    return netR


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



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

        self.process = nn.Sequential(nn.Conv2d(channels * 2, channels // 2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels // 2, channels * 2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(2 * channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        out_instance = self.instance(x1)
        out_identity = x1
        # feature_save(out_identity, '1')
        # feature_save(out_instance, '2')
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)

        out = torch.cat((out1, out2), 1)
        xp1 = self.interative(out) * out2 + out1
        xp2 = (1 - self.interative(out)) * out1 + out2
        xp = torch.cat((xp1, xp2), 1)
        # xp = self.process(self.contrast(xp) + self.avgpool(xp)) * xp
        xp = self.conv1x1(xp)
        xout = xp

        return x1, xout


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
        self.relu = nn.ReLU(True)

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, ref, x_ins):
        ref_x = self.flow_net(ref, x, attn_splits_list=[2, 8],
                              corr_radius_list=[-1, 4],
                              prop_radius_list=[-1, 4],
                              )
        f = ref_x['flow_preds']
        F_warped_move_ = flow_warp2(x_ins, f[-1].permute(0, 2, 3, 1))

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
