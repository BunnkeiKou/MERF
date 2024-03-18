import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.init as init


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


# 多层感知机
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输入特征的维度
        hidden_features = hidden_features or in_features  # 隐藏特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 线性层
        self.drop = nn.Dropout(drop)  # 随机丢弃神经元，丢弃率默认为 0

    # 定义前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 将输入分割为多个不重叠窗口
def window_partition(x, window_size):
    """
    输入:
        x: (B, H, W, C)
        window_size (int): window size  # 窗口的大小

    返回:
        windows: (num_windows*B, window_size, window_size, C)  # 每一个 batch 有单独的 windows
    """
    B, H, W, C = x.shape  # 输入的 batch 个数，高，宽，通道数
    # 将输入 x 重构为结构 [batch 个数，高方向的窗口个数，窗口大小，宽方向的窗口个数，窗口大小，通道数] 的张量
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 交换重构后 x 的第 3和4 维度， 5和6 维度，再次重构为结构 [高和宽方向的窗口个数乘以 batch 个数，窗口大小，窗口大小，通道数] 的张量
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows
    # 这里比较有意思，不太理解的可以给个初始值，比如 x = torch.randn([1, 14, 28, 3])


# 将多个不重叠窗口重新合并
def window_reverse(windows, window_size, H, W):
    """
    输入:
        windows: (num_windows*B, window_size, window_size, C)  # 分割得到的窗口(已处理)
        window_size (int): Window size  # 窗口大小
        H (int): Height of image  # 原分割窗口前特征图的高
        W (int): Width of image  # 原分割窗口前特征图的宽

    返回:
        x: (B, H, W, C)  # 返回与分割前特征图结构一样的结果
    """
    # 以下就是分割窗口的逆向操作，不多解释
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 窗口注意力
class WindowAttention(nn.Module):
    r""" 基于有相对位置偏差的多头自注意力窗口，支持移位的(shifted)或者不移位的(non-shifted)窗口.

    输入:
        dim (int): 输入特征的维度.
        window_size (tuple[int]): 窗口的大小.
        num_heads (int): 注意力头的个数.
        qkv_bias (bool, optional): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        attn_drop (float, optional): 注意力权重的丢弃率，默认为 0.0.
        proj_drop (float, optional): 输出的丢弃率，默认为 0.0.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.window_size = window_size  # 窗口的高 Wh,宽 Ww
        self.num_heads = num_heads  # 注意力头的个数
        head_dim = dim // num_heads  # 每个注意力头对应的通道数
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子 scale，根号d

        # 定义相对位置偏移的参数表，结构为 [2*Wh-1 * 2*Ww-1, num_heads]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取窗口内每个 token 的成对的相对位置索引
        coords_h = torch.arange(self.window_size[0])  # 高维度上的坐标 (0, 7)
        coords_w = torch.arange(self.window_size[1])  # 宽维度上的坐标 (0, 7)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 坐标，结构为 [2, Wh, Ww]
        coords_flatten = torch.flatten(coords, 1)  # 重构张量结构为 [2, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 相对坐标，结构为 [2, Wh*Ww, Wh*Ww]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # 交换维度，结构为 [Wh*Ww, Wh*Ww, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 第1个维度移位
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 第1个维度移位
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1  # 第1个维度的值乘以 2倍的 Ww，再减 1
        relative_position_index = relative_coords.sum(-1)  # 相对位置索引，结构为 [Wh*Ww, Wh*Ww]
        self.register_buffer("relative_position_index", relative_position_index)  # 保存数据，不再更新

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 线性层，特征维度变为原来的 3倍
        self.attn_drop = nn.Dropout(attn_drop)  # 随机丢弃神经元，丢弃率默认为 0.0
        self.proj = nn.Linear(dim, dim)  # 线性层，特征维度不变

        self.proj_drop = nn.Dropout(proj_drop)  # 随机丢弃神经元，丢弃率默认为 0.0

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 截断正态分布，限制标准差为 0.02
        self.softmax = nn.Softmax(dim=-1)  # 激活函数 softmax

    # 定义前向传播
    def forward(self, x, mask=None):
        """
        输入:
            x: 输入特征图，结构为 [num_windows*B, N, C]
            mask: (0/-inf) mask, 结构为 [num_windows, Wh*Ww, Wh*Ww] 或者没有 mask
        """
        B_, N, C = x.shape  # 输入特征图的结构
        # 将特征图的通道维度按照注意力头的个数重新划分，并再做交换维度操作
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 方便后续写代码，重新赋值

        # q 乘以缩放因子
        q = q * self.scale
        # @ 代表常规意义上的矩阵相乘
        attn = (q @ k.transpose(-2, -1))  # q 和 k 相乘后并交换最后两个维度

        # 相对位置偏移，结构为 [Wh*Ww, Wh*Ww, num_heads]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # 相对位置偏移交换维度，结构为 [num_heads, Wh*Ww, Wh*Ww]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)  # 带相对位置偏移的注意力图

        if mask is not None:  # 判断是否有 mask
            nW = mask.shape[0]  # mask 的宽
            # 注意力图与 mask 相加
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)  # 恢复注意力图原来的结构
            attn = self.softmax(attn)  # 激活注意力图 [0, 1] 之间
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # 随机设置注意力图中的部分值为 0
        # 注意力图与 v 相乘得到新的注意力图
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)  # 通过线性层
        x = self.proj_drop(x)  # 随机设置新注意力图中的部分值为 0
        return x


# Swin Transformer 块
class SwinTransformerBlock(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入特征图的分辨率.
        num_heads (int): 注意力头的个数.
        window_size (int): 窗口的大小.
        shift_size (int): SW-MSA 的移位值.
        mlp_ratio (float): 多层感知机隐藏层的维度和嵌入层的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机神经元丢弃率，默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float, optional): 深度随机丢弃率，默认为 0.0.
        act_layer (nn.Module, optional): 激活函数，默认为 nn.GELU.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入特征图的分辨率
        self.num_heads = num_heads  # 注意力头的个数
        self.window_size = window_size  # 窗口的大小
        self.shift_size = shift_size  # SW-MSA 的移位大小
        self.mlp_ratio = mlp_ratio  # 多层感知机隐藏层的维度和嵌入层的比
        if min(self.input_resolution) <= self.window_size:  # 如果输入分辨率小于等于窗口大小
            self.shift_size = 0  # 移位大小为 0
            self.window_size = min(self.input_resolution)  # 窗口大小等于输入分辨率大小
        # 断言移位值必须小于等于窗口的大小
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)  # 归一化层
        # 窗口注意力
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # 如果丢弃率大于 0 则进行随机丢弃，否则进行占位(不做任何操作)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 归一化层
        mlp_hidden_dim = int(dim * mlp_ratio)  # 多层感知机隐藏层维度
        # 多层感知机
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:  # 如果移位值大于 0
            attn_mask = self.calculate_mask(self.input_resolution)  # 计算注意力 mask
        else:
            attn_mask = None  # 注意力 mask 赋空

        self.register_buffer("attn_mask", attn_mask)  # 保存注意力 mask，不参与更新

    # 计算注意力 mask
    def calculate_mask(self, x_size):
        H, W = x_size  # 特征图的高宽
        img_mask = torch.zeros((1, H, W, 1))  # 新建张量，结构为 [1, H, W, 1]
        # 以下两 slices 中的数据是索引，具体缘由尚未搞懂
        h_slices = (slice(0, -self.window_size),  # 索引 0 到索引倒数第 window_size
                    slice(-self.window_size, -self.shift_size),  # 索引倒数第 window_size 到索引倒数第 shift_size
                    slice(-self.shift_size, None))  # 索引倒数第 shift_size 后所有索引
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt  # 将 img_mask 中 h, w 对应索引范围的值置为 cnt
                cnt += 1  # 加 1

        mask_windows = window_partition(img_mask, self.window_size)  # 窗口分割，返回值结构为 [nW, window_size, window_size, 1]
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)  # 重构结构为二维张量，列数为 [window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # 增加第 2 维度减去增加第 3 维度的注意力 mask
        # 用浮点数 -100. 填充注意力 mask 中值不为 0 的元素，再用浮点数 0. 填充注意力 mask 中值为 0 的元素
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    # 定义前向传播
    def forward(self, x, x_size):
        H, W = x_size  # 输入特征图的分辨率
        B, L, C = x.shape  # 输入特征的 batch 个数，长度和维度
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # 归一化
        x = x.view(B, H, W, C)  # 重构 x 为结构 [B, H, W, C]

        # 循环移位
        if self.shift_size > 0:  # 如果移位值大于 0
            # 第 0 维度上移 shift_size 位，第 1 维度左移 shift_size 位
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x  # 不移位

        # 对移位操作得到的特征图分割窗口, nW 是窗口的个数
        x_windows = window_partition(shifted_x, self.window_size)  # 结构为 [nW*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # 结构为 [nW*B, window_size*window_size, C]

        # W-MSA/SW-MSA, 用在分辨率是窗口大小的整数倍的图像上进行测试
        if self.input_resolution == x_size:  # 输入分辨率与设定一致，不需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # 注意力窗口，结构为 [nW*B, window_size*window_size, C]
        else:  # 输入分辨率与设定不一致，需要重新计算注意力 mask
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,
                                         C)  # 结构为 [-1, window_size, window_size, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # 结构为 [B, H', W', C]

        # 逆向循环移位
        if self.shift_size > 0:
            # 第 0 维度下移 shift_size 位，第 1 维度右移 shift_size 位
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x  # 不逆向移位
        x = x.view(B, H * W, C)  # 结构为 [B, H*W， C]

        # FFN
        x = shortcut + self.drop_path(x)  # 对 x 做 dropout，引入残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 归一化后通过 MLP，再做 dropout，引入残差

        return x


# patch 合并，好像没用到
class PatchMerging(nn.Module):
    """
    输入:
        input_resolution (tuple[int]): 输入分辨率.
        dim (int): 输入特征的维度.
        norm_layer (nn.Module, optional): 归一化层，默认 nn.LayerNorm.
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # 输入分辨率
        self.dim = dim  # 输入特征的维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 线性层降维
        self.norm = norm_layer(4 * dim)  # 归一化层

    def forward(self, x):
        """
        x: 结构为 [B, H*W, C]
        """
        H, W = self.input_resolution  # 特征图分辨率的高和宽
        B, L, C = x.shape  # 输入 x 的特征batch，长度和通道数，要注意这个 B
        # 断言输入的长度等于 H*W，则输入图片大小不对
        assert L == H * W, "input feature has wrong size"
        # 断言 H 和 W 是偶数
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)  # 重构 x 为 [B, H, W, C]
        # 主要是对数据原始结构的理解，反正我感觉这块看不懂
        x0 = x[:, 0::2, 0::2, :]  # 结构为 [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # 结构为 [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # 结构为 [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # 结构为 [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # 结构为 [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # 结构为 [B, H/2*W/2, 4*C]

        x = self.norm(x)  # 归一化
        x = self.reduction(x)  # 线性层降维

        return x


# 单阶段的 SWin Transformer 基础层
class BasicLayer(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        depth (int): SWin Transformer 块的个数.
        num_heads (int): 注意力头的个数.
        window_size (int): 本地(当前块中)窗口的大小.
        mlp_ratio (float): MLP隐藏层特征维度与嵌入层特征维度的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入分辨率
        self.depth = depth  # SWin Transformer 块的个数
        self.use_checkpoint = use_checkpoint  # 是否使用 checkpointing 来节省显存，默认为 False

        # 创建 Swin Transformer 网络
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,  # MSA位移为0，W-MSA位移为1/2的windowsize
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch 合并层
        if downsample is not None:  # 如果有下采样
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)  # 下采样
        else:
            self.downsample = None  # 不做下采样

    # 定义前向传播
    def forward(self, x, x_size):
        for blk in self.blocks:  # x 输入串联的 Swin Transformer 块
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)  # 使用 checkpoint
            else:
                x = blk(x, x_size)  # 直接输入网络
        if self.downsample is not None:
            x = self.downsample(x)  # 下采样
        return x


# 残差 Swin Transforme 块 (RSTB)
class RSTB(nn.Module):
    """
    输入:
        dim (int): 输入特征的维度.
        input_resolution (tuple[int]): 输入分辨率.
        depth (int): SWin Transformer 块的个数.
        num_heads (int): 注意力头的个数.
        window_size (int): 本地(当前块中)窗口的大小.
        mlp_ratio (float): MLP隐藏层特征维度与嵌入层特征维度的比.
        qkv_bias (bool, optional): 给 query, key, value 添加一个可学习偏置，默认为 True.
        qk_scale (float | None, optional): 重写默认的缩放因子 scale.
        drop (float, optional): D 随机丢弃神经元，丢弃率默认为 0.0.
        attn_drop (float, optional): 注意力图随机丢弃率，默认为 0.0.
        drop_path (float | tuple[float], optional): 深度随机丢弃率，默认为 0.0.
        norm_layer (nn.Module, optional): 归一化操作，默认为 nn.LayerNorm.
        downsample (nn.Module | None, optional): 结尾处的下采样层，默认没有.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
        img_size: 输入图片的大小.
        patch_size: Patch 的大小.
        resi_connection: 残差连接之前的卷积块.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim  # 输入特征的维度
        self.input_resolution = input_resolution  # 输入分辨率

        # SWin Transformer 基础层
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':  # 结尾用 1 个卷积层
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':  # 结尾用 3 个卷积层
            # 为了减少参数使用和节约显存，采用瓶颈结构
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # 图像转成 Patch Embeddings
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        # 从 Patch Embeddings 组合图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    # 定义前向传播
    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x  # 引入残差


# 图像转成 Patch Embeddings
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)  # 归一化
        else:
            self.norm = None

    # 定义前向传播
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C] # [B, HW, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


# 从 Patch Embeddings 组合图像
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # 输出结构为 [B, embed_dim, H, W]
        return x


# 上采样
class Upsample(nn.Sequential):
    """
    输入:
        scale (int): 缩放因子，支持 2^n and 3.
        num_feat (int): 中间特征的通道数.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # 缩放因子等于 2^n
            for _ in range(int(math.log(scale, 2))):  # 循环 n 次
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))  # 卷积层
                m.append(nn.PixelShuffle(2))  # pixelshuffle 上采样 2 倍
        elif scale == 3:  # 缩放因子等于 3
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))  # 卷积层
            m.append(nn.PixelShuffle(3))  # pixelshuffle 上采样 3 倍
        else:
            # 报错，缩放因子不对
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# 一步实现既上采样也降维
class UpsampleOneStep(nn.Sequential):
    """一步上采样与前边上采样模块不同之处在于该模块只有一个卷积层和一个 pixelshuffle 层

    输入:
        scale (int): 缩放因子，支持 2^n and 3.
        num_feat (int): 中间特征的通道数.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat  # 中间特征的通道数
        self.input_resolution = input_resolution  # 输入分辨率
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))  # 卷积层
        m.append(nn.PixelShuffle(scale))  # pixelshuffle 上采样 scale 倍
        super(UpsampleOneStep, self).__init__(*m)


def srntt_init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


# SWinIR
class SwinIR(nn.Module):
    r""" SwinIR
        基于 Swin Transformer 的图像恢复网络.

    输入:
        img_size (int | tuple(int)): 输入图像的大小，默认为 64*64.
        patch_size (int | tuple(int)): patch 的大小，默认为 1.  就是下采样的倍数
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): Patch embedding 的维度，默认为 96.
        depths (tuple(int)): Swin Transformer 层的深度. depth的list长度就是RSTB的个数，每一个list的数量就是STL块的数量
        num_heads (tuple(int)): 在不同层注意力头的个数.
        window_size (int): 窗口大小，默认为 7.
        mlp_ratio (float): MLP隐藏层特征图通道与嵌入层特征图通道的比，默认为 4.
        qkv_bias (bool): 给 query, key, value 添加可学习的偏置，默认为 True.
        qk_scale (float): 重写默认的缩放因子，默认为 None.
        drop_rate (float): 随机丢弃神经元，丢弃率默认为 0.
        attn_drop_rate (float): 注意力权重的丢弃率，默认为 0.
        drop_path_rate (float): 深度随机丢弃率，默认为 0.1.
        norm_layer (nn.Module): 归一化操作，默认为 nn.LayerNorm.
        ape (bool): patch embedding 添加绝对位置 embedding，默认为 False.
        patch_norm (bool): 在 patch embedding 后添加归一化操作，默认为 True.
        use_checkpoint (bool): 是否使用 checkpointing 来节省显存，默认为 False.
        upscale: 放大因子， 2/3/4/8 适合图像超分, 1 适合图像去噪和 JPEG 压缩去伪影
        img_range: 灰度值范围， 1 或者 255.
        upsampler: 图像重建方法的选择模块，可选择 pixelshuffle, pixelshuffledirect, nearest+conv 或 None.
        resi_connection: 残差连接之前的卷积块， 可选择 1conv 或 3conv.
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans  # 输入图片通道数
        num_out_ch = 3  # 输出图片通道数
        num_feat = 64  # 特征图通道数
        self.img_range = img_range  # 灰度值范围:[0, 1] or [0, 255]

        self.upscale = upscale  # 图像放大倍数，超分(2/3/4/8),去噪(1)
        self.upsampler = upsampler  # 上采样方法
        self.window_size = window_size  # 注意力窗口的大小

        ################################### 1, 浅层特征提取 ###################################
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.pad = nn.ReflectionPad2d(1)
        self.conv_first = nn.Conv2d(64 * 2, embed_dim, kernel_size=3)  # 输入卷积层

        ################################### 2, 深层特征提取 ######################################
        self.num_layers = len(depths)  # Swin Transformer 层的个数 4层
        self.embed_dim = embed_dim  # 嵌入层特征图的通道数
        self.ape = ape  # patch embedding 添加绝对位置 embedding，默认为 False.
        self.patch_norm = patch_norm  # 在 patch embedding 后添加归一化操作，默认为 True.
        self.num_features = embed_dim  # 特征图的通道数
        self.mlp_ratio = mlp_ratio  # MLP隐藏层特征图通道与嵌入层特征图通道的比

        # 将图像维度变成1维
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)  # 截断正态分布，限制标准差为0.02

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.pooling = nn.AvgPool2d(2, 2)
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.after = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        srntt_init_weights(self, init_type='normal', init_gain=0.02)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, x_ref):

        x = self.check_image_size(x)
        x_ref = self.check_image_size(x_ref)

        x_t = self.conv_first(self.pad(torch.cat([x, x_ref], dim=1)))
        x_t = self.conv_after_body(self.forward_features(x_t))

        out = self.after(x_t)
        return out, x_t
