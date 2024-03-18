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

import scipy
import kornia

import torch.nn.functional as F


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def norm(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = numpy.zeros(numpy.shape(data))
    m = data.shape[0]
    normData = data - minVals
    normData = normData / ranges
    return normData


def visualize_feature_map(img_batch, path, color=True):
    # feature_map = numpy.squeeze(img_batch, axis=0)
    feature_map = img_batch
    print(feature_map.shape)

    feature_map_combination = []

    num_pic = feature_map.shape[2]
    # row, col = get_row_col(num_pic)

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        # plt.imshow(feature_map_split)
        # plt.axis('off')
        # plt.title('feature_map_{}'.format(i))

    # plt.savefig('feature_map.png')
    # plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    feature_map = norm(feature_map_sum)
    image_numpy = feature_map * 255.0
    outputimage = Image.fromarray(numpy.uint8(image_numpy))
    outputimage.save(path)

    if color:
        im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cv2.imwrite(path, im_color)


def visualize_feature_map2(img_batch, path, pick=0, color=True):
    # feature_map = numpy.squeeze(img_batch, axis=0)
    feature_map = img_batch
    print(feature_map.shape)
    # feature_map_combination = []

    # num_pic = feature_map.shape[2]
    # row, col = get_row_col(num_pic)
    feature_map = feature_map[:, :, pick]
    feature_map = norm(feature_map)
    image_numpy = feature_map * 255.0
    outputimage = Image.fromarray(numpy.uint8(image_numpy))
    outputimage.save(path)

    if color:
        im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        cv2.imwrite(path, im_color)


def fuse_Cx(A, B):
    A = A * 255
    B = B * 255
    A_128 = abs(A - 128)
    B_128 = abs(B - 128)
    fuse = (A * A_128 + B * B_128) / (A_128 + B_128)
    fuse[torch.isnan(fuse)] = 128
    return fuse / 255.0


class Gray(object):
    def __call__(self, tensor):
        R = tensor[:, 0, :, :]
        G = tensor[:, 1, :, :]
        B = tensor[:, 2, :, :]
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        gray = torch.unsqueeze(gray, dim=1)
        return gray


def select_ref_idx(seq_imgs, win_size=None, expos_thres=None):
    if win_size == None:
        win_size = 3

    if expos_thres == None:
        expos_thres = 0.01

    seq_imgs = numpy.double(seq_imgs)
    # seq_imgs = reorder_by_lum(seq_imgs)
    [_, _, size_3, size_4] = seq_imgs.shape
    window = numpy.ones((win_size, win_size, 3))
    window = window / window.sum()
    positive = numpy.zeros((size_4, 1))  # size_4 是图片序列数量
    for i in range(size_4):
        conved_img = scipy.signal.convolve(seq_imgs[:, :, :, i], window, 'valid')
        positive[i] = numpy.sum(numpy.sum((conved_img < expos_thres) | (conved_img > 1 - expos_thres)))
    ref_idx = numpy.argmin(positive)  # 最小值对应的索引
    return ref_idx


def gradient_im(im):
    sobel_kernel = torch.Tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
    sobel_kernel = torch.reshape(sobel_kernel, ((1, 1, 3, 3)))
    sobel_kernel = sobel_kernel.cuda()  # 加载到GPU上运算
    weight = sobel_kernel
    gradient_im = F.conv2d(im, weight, stride=1, padding=1)
    # edge_detect = edge_detect.squeeze().detach().cpu().numpy()
    return gradient_im


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
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    # dataset_size = len(data_loader)
    model = create_model(opt)
    # dataset = data_loader.load_data()
    # dataset_size = len(data_loader)
    skip = True if opt.skip > 0 else False
    # 先新建空模型，后面load_network来加载模型中的参数
    # model.denoiser = network_dncnn.DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R')
    # model.denoiser.cuda()
    # model.denoiser.load_state_dict(torch.load("./model/dncnn_color_blind.pth"), strict=True)
    model.netA = define_A(opt, test_mode=True)

    model.Mapping = networks.define_Att()
    model.netD = networks.define_F()
    # model.Mapping.load_state_dict(torch.load("./checkpoints/Ex_S/400_net_M.pth"), strict=True)
    # model.Mapping.eval()
    # model.correct_high = networks.define_Att()
    # model.netG.cuda()
    model.netG = define_G(window_size=4)
    # # model.netG.load_state_dict(torch.load("./checkpoints/static/400_net_G.pth"), strict=True)
    # model.netG.eval()
    # # model.netG_A_high = BasicSpatialAttentionNet()
    # # model.color_net = networks.define_GC(model.gpu_ids, skip=skip, opt=opt)
    # # model.color_net.load_state_dict(torch.load("checkpoints/DEM+CEM(L1+0.1vgg)/200_net_G_color.pth"), strict=True)
    model.refinement_net = networks.define_R(model.gpu_ids)
    # # model.refinement_net.load_state_dict(torch.load("./checkpoints/static/400_net_R.pth"), strict=True)
    # model.refinement_net.eval()
    # model.refinement_net.cuda()
    print("---is not train----")
    # which_epoch = opt.which_epoch
    print("---model is loaded---")
    # model.netG_A_low.cpu()
    # model.netG_A_high.cpu()
    # model.color_net.cpu()
    # 将模型保存路径中的G_A,G_V加载到model对应网络中
    # model.Mapping.load_state_dict(torch.load("./checkpoints/Ex_S/400_net_M.pth"), strict=True)
    model.load_network(model.netA, 'A', 400)
    model.load_network(model.netD, 'D', 400)
    model.load_network(model.Mapping, 'M', 400)
    model.load_network(model.netG, 'G', 400)
    model.load_network(model.refinement_net, 'R', 400)
    # flow_net = torch.nn.DataParallel(RAFT(opt))
    # flow_net.cuda()
    # flow_net.eval()

    # model.netG.load_state_dict(torch.load("./checkpoints/Ex_S/400_net_G.pth"), strict=True)
    # # model.load_network(model.correct_high, 'H', 400)
    # model.refinement_net.load_state_dict(torch.load("./checkpoints/Ex_S/400_net_R.pth"), strict=True)
    print('---------- Networks initialized -------------')
    # networks.print_network(model.netG_A)
    # networks.print_network(model.d_net)
    # networks.print_network(model.h_net)
    model.netA.eval()
    model.netG.eval()
    model.netD.eval()
    # # model.correct_Low.eval()
    model.Mapping.eval()
    model.refinement_net.eval()
    # #
    # model.denoiser.eval()
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
        # imgA_path = "/data/H/Robust/low/" + str(i) + ".png"
        # imgB_path = "/data/H/Robust/high/" + str(i) + ".png"
        # imgA_path = "/data/H/extreme/trainA/" + str(i) + ".png"
        # if not os.path.exists(imgA_path):
        #     imgA_path = "/data/H/extreme/trainA/" + str(i) + ".PNG"
        #
        # imgB_path = "/data/H/extreme/trainB/" + str(i) + ".png"
        # if not os.path.exists(imgA_path):
        #     imgB_path = "/data/H/extreme/trainB/" + str(i) + ".PNG"

        # img_A_correct = "fused_results_test100/" + str(opt.name) + "/A/" + str(i).zfill(3) + ".png"
        # img_B_correct = "fused_results_test100/" + str(opt.name) + "/B/" + str(i).zfill(3) + ".png"
        # img_identity_A = "fused_results_test100/" + str(opt.name) + "/A_i/" + str(i).zfill(3) + ".png"
        # img_identity_B = "fused_results_test100/" + str(opt.name) + "/B_i/" + str(i).zfill(3) + ".png"
        # img_B_correct = "fused_results_test100/" + str(opt.name) + "/B/" + str(i).zfill(3) + ".png"

        # lf_img = "fused_results_test100/" + str(opt.name) + "-v/lf/" + str(i) + ".png"
        #
        # warp_refine = "fused_results_test100/" + str(opt.name) + "-v/warp_refine/" + str(i) + ".png"
        output_path = "fused_results_test100/" + str(opt.name) + "-v/output/" + str(i) + ".png"

        try:
            imgA = cv2.imread(imgA_path)
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
            H, W, C = imgA.shape
            imgA = cv2.resize(imgA, (W - W % 32, H - H % 32))
            imgB = cv2.imread(imgB_path)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
            imgB = cv2.resize(imgB, (W - W % 32, H - H % 32))
            # imgB = cv2.resize(imgB, (320, 320))
            # imgC = cv2.imread(imgC_path)
            # imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)

            total_num += 1
        except:
            continue

        imgA = double(imgA) / 255
        imgB = double(imgB) / 255
        # imgC = double(imgC)/255

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
            # align_A_high = align_A - up(align_A_half)
            # align_B_high = align_B - up(align_B_half)
            detail = model.refinement_net.forward(align_B, align_A, x_lf)
            refinement = detail + o_lf

        # correct_A = correct_A.cpu()
        # correct_A = correct_A.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # visualize_feature_map2(correct_A, img_A_correct, 2)
        #
        # correct_B = correct_B.cpu()
        # correct_B = correct_B.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # visualize_feature_map2(correct_B, img_B_correct, 2)
        #
        # identity_A = identity_A.cpu()
        # identity_A = identity_A.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # visualize_feature_map2(identity_A, img_identity_A, 2)
        #
        # identity_B = identity_B.cpu()
        # identity_B = identity_B.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # visualize_feature_map2(identity_B, img_identity_B, 2)

        # img_warp_refine = img_warp.cpu()
        # img_warp_refine_numpy = img_warp_refine.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # img_warp_refine_numpy = img_warp_refine_numpy * 255.0
        # img_warp_refine_numpy = numpy.maximum(img_warp_refine_numpy, 0)
        # img_warp_refine_numpy = numpy.minimum(img_warp_refine_numpy, 255)
        # img_warp_refine_image = Image.fromarray(numpy.uint8(img_warp_refine_numpy))
        # img_warp_refine_image.save(warp_refine)
        #
        # o_lf = o_lf.cpu()
        # o_lf_numpy = o_lf.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        # o_lf_numpy = o_lf_numpy * 255.0
        # o_lf_numpy = numpy.maximum(o_lf_numpy, 0)
        # o_lf_numpy = numpy.minimum(o_lf_numpy, 255)
        # o_lf_image = Image.fromarray(numpy.uint8(o_lf_numpy))
        # o_lf_image.save(lf_img)

        output = refinement.cpu()
        output_numpy = output.permute(0, 3, 2, 1).squeeze(0).detach().numpy()
        output_numpy = output_numpy * 255.0
        output_numpy = numpy.maximum(output_numpy, 0)
        output_numpy = numpy.minimum(output_numpy, 255)
        output_numpy = Image.fromarray(numpy.uint8(output_numpy))
        output_numpy.save(output_path)

        print("ok")
