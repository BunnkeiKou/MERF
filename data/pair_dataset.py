import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

from PIL import Image
import PIL
import random
import torch
from pdb import set_trace as st
import numpy as np
import cv2
import time
import scipy


class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'low')
        self.dir_B = os.path.join(opt.dataroot, 'high')
        self.dir_B2 = os.path.join(opt.dataroot, 'align')
        self.dir_C = os.path.join(opt.dataroot, 'gt')
        self.dir_flow = os.path.join(opt.dataroot, 'flows')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.B2_paths = make_dataset(self.dir_B2)
        # self.E_paths = make_dataset(self.dir_E)
        self.C_paths = make_dataset(self.dir_C)
        self.flow_paths = make_dataset(self.dir_flow)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.B2_paths = sorted(self.B2_paths)
        # self.E_paths = sorted(self.E_paths)
        self.C_paths = sorted(self.C_paths)
        self.flow_paths = sorted(self.flow_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.B2_size = len(self.B2_paths)
        # self.E_size = len(self.E_paths)
        self.C_size = len(self.C_paths)
        self.F_size = len(self.flow_paths)

        transform_list = []

        transform_list += [transforms.ToTensor()]
        # 函数接受PIL Image或numpy.ndarray，将其先由HWC转置为CHW格式，再转为float后每个像素除以255
        # transform_list = [transforms.ToTensor()]

        self.transform1 = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
            transforms.ToTensor()
        ])

        self.transform = transforms.Compose(transform_list)
        # self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        B2_path = self.B2_paths[index % self.B2_size]
        # E_path = self.B_paths[index % self.E_size]
        C_path = self.C_paths[index % self.C_size]
        flow_path = self.flow_paths[index % self.A_size]

        # print(A_path)
        A_img = Image.open(A_path).convert('RGB')
        # a = Image.open(A_path).convert('RGB')
        B2_img = Image.open(A_path.replace("low", "align")).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # b = Image.open(A_path.replace("low", "high").replace("A", "B")).convert('RGB')
        C_img = Image.open(A_path.replace("low", "gt")).convert('RGB')
        flows = np.load(flow_path)
        flows = torch.from_numpy(flows).squeeze(0)
        # E_img = Image.open(A_path.replace("low", "mid").replace("A", "E")).convert('RGB')
        # print("kkkkkkkk:",len(C_img.split()))

        A_img = self.transform(A_img)  # C H W
        # a = self.transform(a)
        B2_img = self.transform(B2_img)
        B_img = self.transform(B_img)
        # b = self.transform(b)
        # aaa = C_img
        C_img = self.transform(C_img)
        # C_d = self.transform1(aaa)
        # E_img = self.transform(E_img)

        # select ref
        # A_img = u_img.permute(1, 2, 0).unsqueeze(3)
        # B_img = o_img.permute(1, 2, 0).unsqueeze(3)
        # input_seq = torch.cat([a, b], dim=3)
        # index = select_ref_idx(input_seq)
        # if index == 1:
        #     A_img = o_img
        #     B_img = u_img
        # else:
        #     A_img = u_img
        #     B_img = o_img

        # if self.opt.is_haze:
        #     A_img = -A_img
        #     B_img = -B_img

        w = A_img.size(2)
        h = A_img.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A_img = A_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        # a = a[:, h_offset:h_offset + self.opt.fineSize,
        #     w_offset:w_offset + self.opt.fineSize]
        B2_img = B2_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        B_img = B_img[:, h_offset:h_offset + self.opt.fineSize,
                 w_offset:w_offset + self.opt.fineSize]
        # b = b[:, h_offset:h_offset + self.opt.fineSize,
        #     w_offset:w_offset + self.opt.fineSize]
        C_img = C_img[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        flows = flows[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]
        # C_d = C_d[:, h_offset:h_offset + self.opt.fineSize,
        #       w_offset:w_offset + self.opt.fineSize]

        if self.opt.resize_or_crop == 'no':
            r, g, b = A_img[0] + 1, A_img[1] + 1, A_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                # a = a.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
                # b = b.index_select(2, idx)
                C_img = C_img.index_select(2, idx)
                # C_d = C_d.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                # a = a.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
                # b = b.index_select(1, idx)
                C_img = C_img.index_select(1, idx)
                # C_d = C_d.index_select(1, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
                input_img = (A_img + 1) / 2. / times
                input_img = input_img * 2 - 1
            else:
                input_img = A_img
            r, g, b = input_img[0] + 1, input_img[1] + 1, input_img[2] + 1
            A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
            A_gray = torch.unsqueeze(A_gray, 0)

        return {'A': A_img, 'B': B_img, 'B2': B2_img, 'C': C_img, 'A_paths': A_path, 'B_paths': B_path,
                'C_paths': C_path, 'flow': flows}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'PairDataset'
