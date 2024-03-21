import torch
from torch import nn
from torch.autograd import Variable
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import numpy as np
from math import exp


def latent2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


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

        self.avg_pool = nn.AvgPool2d(2, 2)

        self.netA = networks.define_A(opt)
        # self.netA.load_state_dict(torch.load("./checkpoints/Final_Flow/800_net_A.pth"), strict=True)
        window_size = 4
        self.Mapping = networks.define_Att()
        self.Dual_Att = networks.define_F()

        # self.Mapping.load_state_dict(torch.load("./checkpoints/Final_Flow/800_net_M.pth"), strict=True)

        self.netG = networks.define_G(gpu_ids=[], window_size=window_size)
        # self.netG.load_state_dict(torch.load("./checkpoints/Ex_S_SICE/400_net_G.pth"), strict=True)

        self.refinement_net = networks.define_R()
        # self.refinement_net.load_state_dict(torch.load("./checkpoints/Ex_S_SICE/400_net_R.pth"), strict=True)

        which_epoch = 400
        self.load_network(self.netG, 'G', which_epoch)
        self.load_network(self.Dual_Att, 'D', which_epoch)
        self.load_network(self.Mapping, 'M', which_epoch)
        self.load_network(self.netA, 'A', which_epoch)
        self.load_network(self.refinement_net, 'R', which_epoch)

        print('---------- Networks initialized -------------')

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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def predict(self):
        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        with torch.no_grad():
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
            output = latent2im(self.refinement.data)
        return output
