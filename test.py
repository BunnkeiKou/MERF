import os.path
import numpy
import torch
from numpy import double
from models import networks
from options.train_options import TrainOptions
from models.models import create_model
from PIL import Image
import cv2
from models.networks import define_G, define_A
from torch import nn
import kornia
import torch.nn.functional as F


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)


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
    # flow_x = flow[:, :, :, 0] * h / 320
    # flow_y = flow[:, :, :, 1] * w / 320
    # flow = torch.cat((flow_x.unsqueeze(3), flow_y.unsqueeze(3)), dim=3)
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


if __name__ == '__main__':
    opt = TrainOptions().parse()
    config = get_config(opt.config)
    model = create_model(opt)

    skip = True if opt.skip > 0 else False
    model.netA = define_A(opt, test_mode=True)

    model.Mapping = networks.define_Att()
    model.netD = networks.define_F()
    model.netG = define_G(window_size=4)
    model.refinement_net = networks.define_R(model.gpu_ids)

    print("---is not train----")

    print("---model is loaded---")

    model.load_network(model.netA, 'A', 400)
    model.load_network(model.netD, 'D', 400)
    model.load_network(model.Mapping, 'M', 400)
    model.load_network(model.netG, 'G', 400)
    model.load_network(model.refinement_net, 'R', 400)

    print('---------- Networks initialized -------------')

    model.netA.eval()
    model.netG.eval()
    model.netD.eval()

    model.Mapping.eval()
    model.refinement_net.eval()

    total_psnr = 0
    total_ssim = 0
    total_num = 0

    # mkdir("./fused_results_test100/" + str(opt.name) + "-v/A/")
    # mkdir("./fused_results_test100/" + str(opt.name) + "-v/B/")

    # mkdir("./fused_results_test100/" + str(opt.name) + "-v/warp_refine/")
    mkdir("./fused_results_test100/" + str(opt.name) + "-v/output/")
    # mkdir("./fused_results_test100/" + str(opt.name) + "-v/lf/")
    # mkdir("./fused_results_test100/" + str(opt.name) + "/A_i/")
    # mkdir("./fused_results_test100/" + str(opt.name) + "/B_i/")

    for i in range(1, 151):
        imgA_path = "/data/H/Valid/low/low" + str(i).zfill(5) + ".png"
        imgB_path = "/data/H/Valid/high/high" + str(i).zfill(5) + ".png"

        output_path = "fused_results_test100/" + str(opt.name) + "-v/output/" + str(i) + ".png"

        try:
            imgA = cv2.imread(imgA_path)
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
            H, W, C = imgA.shape
            imgA = cv2.resize(imgA, (W - W % 32, H - H % 32))
            imgB = cv2.imread(imgB_path)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
            imgB = cv2.resize(imgB, (W - W % 32, H - H % 32))

            total_num += 1
        except:
            continue

        imgA = double(imgA) / 255
        imgB = double(imgB) / 255


        imgA = torch.from_numpy(imgA)
        imgB = torch.from_numpy(imgB)

        imgA = imgA.unsqueeze(0)
        imgB = imgB.unsqueeze(0)

        imgA = imgA.permute(0, 3, 2, 1).float().cuda()
        imgB = imgB.permute(0, 3, 2, 1).float().cuda()

        up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        avg_pool = nn.AvgPool2d(2, 2)
        with torch.no_grad():
            identity_A, correct_A = model.Mapping.forward(imgA)
            identity_B, correct_B = model.Mapping.forward(imgB)
            align_B, f_w = model.netA.forward(correct_B, correct_A, identity_B)
            align_B, align_A = model.netD.forward(align_B, identity_A)
            img_warp = flow_warp2(imgB, f_w[-1].permute(0, 2, 3, 1))
            align_B_half = avg_pool(align_B)
            align_A_half = avg_pool(align_A)
            o_lf, x_lf = model.netG.forward(align_B_half, align_A_half)
            o_lf = up(o_lf)
            x_lf = up(x_lf)

            detail = model.refinement_net.forward(align_B, align_A, x_lf)
            refinement = detail + o_lf
        output = refinement.cpu()
        output_numpy = output.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        output_numpy = output_numpy * 255.0
        output_numpy = numpy.maximum(output_numpy, 0)
        output_numpy = numpy.minimum(output_numpy, 255)
        output_numpy = Image.fromarray(numpy.uint8(output_numpy))
        output_numpy.save(output_path)

        print("ok")
