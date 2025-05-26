import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import cv2
from thop import profile

from einops import rearrange

from modules.BasicBlock import *


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False,
                 relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        stride = kernel_size // 2
        self.conv1 = BasicConv(n_feat, n_feat, kernel_size, stride, bias=bias)
        self.conv2 = BasicConv(n_feat, 1, kernel_size, stride, bias=bias)
        self.conv3 = BasicConv(1, n_feat, kernel_size, stride, bias=bias)

    def forward(self, x, x_img):
        x_img = x_img.to('cuda')
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class ResBlock_do(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock_do, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward', att=False):
        super(ResBlock_do_fft_bench, self).__init__()
        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.att = att
        if self.att:
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (32, 128)])
            # self.at = CALayer(out_channel, reduction=8)
            self.att = MultiSpectralAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel], reduction=16,
                                                   freq_sel_method='top16')
        self.dim = out_channel
        self.norm = norm
        self.c = 2
        #大小为256*256
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, 129))
        #大小为128*128
        # self.sqrt_beta = nn.Parameter(torch.randn(1, 1, 65))

    def forward(self, x):
        dim = 1
        
        k = x[1]
        #k=0.9时c=1，k=0.7或0.5时c=2，k=0.3时c=3
        if k==0.9:
            self.c = 1
        elif k==0.7 or k==0.5:
            self.c = 2
        else:
            self.c = 3
        x = x[0]
        _,batch, seq_len, hidden = x.shape
        y = torch.fft.rfft(x, norm='ortho')
        low_pass = y[:]
        low_pass[:, self.c:, :] = 0
        
        high_pass = y - low_pass
        y = (1-self.sqrt_beta) * low_pass + self.sqrt_beta * high_pass


        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft(y, n=seq_len, norm='ortho')
        
        '''
        low_pass = torch.fft.irfft(low_pass, n=seq_len, norm='ortho')
        high_pass = x - low_pass
        low_pass = self.main(low_pass)
        y = low_pass + (self.sqrt_beta**2) * high_pass
        '''
        '''
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        '''
        if self.att:
            yy = self.main(x)
            ca = self.att(yy)
            out = ca + x + y
        else:
            out = self.main(x) + x + y
        return [out, k]


def window_partitions(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows

def dynamic_prompt_K(x, key):
    # 将x转为numpy数组
    x = x.cpu().detach().numpy()
    # 根据阈值分割云和非云像素
    _, binary_mask = cv2.threshold(x, 200, 255, cv2.THRESH_BINARY)
    # 计算云掩模占整个图像的百分比
    cloud_ratio = np.count_nonzero(binary_mask) / binary_mask.size

    # 根据云掩模的比例返回不同的K值
    if cloud_ratio < 0.15:
        K = 0.9
    elif cloud_ratio < 0.25:
        K = 0.7
    elif cloud_ratio < 0.35:
        K = 0.5
    else:
        K = 0.3

    # 为key关联的云比例区间创建映射
    thresholds = {
        3: [0.05, 0.10, 0.15],
        2: [0.10, 0.15, 0.20]
    }

    # 根据key获取对应的阈值列表
    if key in thresholds:
        thresholds_list = thresholds[key]
        num_modules = sum(cloud_ratio >= t for t in thresholds_list)
    else:
        num_modules = 1

    return K, num_modules


def window_reverses(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # print('B: ', B)
    # print(H // window_size)
    # print(W // window_size)
    C = windows.shape[1]
    # print('C: ', C)
    x = windows.view(-1, H // window_size, W // window_size, C, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return x


def window_partitionx(x, window_size):
    _, _, H, W = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partitions(x[:, :, :h, :w], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, :, -window_size:, -window_size:]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partitions(x[:, :, :h, -window_size:], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partitions(x[:, :, -window_size:, :w], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane - 1, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = x.to('cuda')
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


def window_reversex(windows, window_size, H, W, batch_list):
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_reverses(windows[:batch_list[0], ...], window_size, h, w)
    B, C, _, _ = x_main.shape
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    res = torch.zeros([B, C, H, W], device=windows.device)
    res[:, :, :h, :w] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverses(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, :, h:, w:] = x_dd[:, :, h - H:, w - W:]
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
        x_d = window_reverses(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :, :h, w:] = x_r[:, :, :, w - W:]
    if h != H and len(batch_list) == 2:
        x_d = window_reverses(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, :, h:, :w] = x_d[:, :, h - H:, :]
    return res


class ResBlock(nn.Module):
    def __init__(self, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True, norm=False),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False, norm=False)
        )

    def forward(self, x):
        return self.main(x) + x


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, out_channel, num_res, ResBlock=ResBlock):
        super(DBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF_att_Fca(nn.Module):
    def __init__(self, in_channel, out_channel, att=True):
        super(AFF_att_Fca, self).__init__()
        self.att = att
        self.conv1 = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True, relu_method=nn.LeakyReLU),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        if self.att:
            c2wh = dict([(64, 56), (128, 28), (256, 14), (512, 7), (32, 128)])
            # self.att = CALayer(out_channel, reduction=8)
            self.attention = MultiSpectralAttentionLayer(out_channel, c2wh[out_channel], c2wh[out_channel],
                                                         reduction=16, freq_sel_method='low16')
        # self.conv = BasicConv(out_channel * 2, out_channel * 2, kernel_size=1, stride=1,
        #                       relu=True, relu_method=nn.LeakyReLU)

    def forward(self, x, up):
        y1 = torch.cat([x, up], dim=1)
        y2 = self.conv1(y1)
        att = self.attention(y2)
        return torch.cat([att + x, up], dim=1)


class FCN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(out_channel, out_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(out_channel, out_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(out_channel, out_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(out_channel, out_channel, kernel_size=3, relu=True, stride=1)
        )

    def forward(self, x):
        return self.fcn(x)


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        # 对其他涉及的张量也进行相同处理
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x,K):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        #取出每个head中的前C*K个最大值的索引
        index = torch.topk(attn, k=int(C*K), dim=-1, largest=True)[1]
        #将mask中的index位置置为1
        mask = mask.scatter(-1, index, 1)
        #在attn中将mask中为1的位置置为-inf
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
        
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class Attention_Qeury(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Qeury, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature, K):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = self.q_dwconv(torch.cat([q, feature], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        #取出每个head中的前C*K个最大值的索引
        index = torch.topk(attn, k=int(C*K), dim=-1, largest=True)[1]
        #将mask中的index位置置为1
        mask = mask.scatter(-1, index, 1)
        #在attn中将mask中为1的位置置为-inf
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Attention_Key_Value(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_Key_Value, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

        self.v_dwconv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2, K):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        k = self.k_dwconv(torch.cat([k, feature1], dim=1))
        v = self.v_dwconv(torch.cat([v, feature2], dim=1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        #取出每个head中的前C*K个最大值的索引
        index = torch.topk(attn, k=int(C*K), dim=-1, largest=True)[1]
        #将mask中的index位置置为1
        mask = mask.scatter(-1, index, 1)
        #在attn中将mask中为1的位置置为-inf
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        K = x[1]
        x = x[0]
        x = x + self.attn(self.norm1(x),K)
        x = x + self.ffn(self.norm2(x))

        return [x,K]


class TransformerBlock_QKV(nn.Module):
    def __init__(self, dim=32, num_heads=1, ffn_expansion_factor=3, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_QKV, self).__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn_Qeury = Attention_Qeury(dim, num_heads, bias)
        self.attn_Key_Value = Attention_Key_Value(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.fusion = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=2 * dim, bias=bias), nn.GELU(),
            nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias))

    def forward(self, x, feature1, feature2, feature3):
        # print('dim',self.dim)
        #x, feature1, feature2, feature3 = input[0], input[1], input[2], input[3]
        K = x[1]
        x = x[0]
        attn_Qeury = self.attn_Qeury(self.norm1(x), feature1, K)
        attn_KV = self.attn_Key_Value(self.norm1(x), feature2, feature3, K)

        x = x + self.fusion(torch.cat([attn_Qeury, attn_KV], dim=1))
        x = x + self.ffn(self.norm2(x))

        return [x,K]

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################

class DPM(nn.Module):
    def __init__(self,dim,num_heads,ffn_expansion_factor,bias,LayerNorm_type,dual_pixel_task=False):
        super(DPM, self).__init__()
        #global attention
        self.qkv_dwconv = nn.Sequential(nn.Conv2d(dim , dim * 3, kernel_size=1, stride=1),
                                        nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim,
                                                    bias=True))
        
        self.Attention_qkv = TransformerBlock_QKV(dim, num_heads=num_heads,ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.layer_norm = LayerNorm(dim,LayerNorm_type)
        #local attention
        self.degradation = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1),
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias))
         ###############################
        self.input = nn.Sequential(
                nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
            )

        self.main_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

        self.degradation_kernel = nn.Sequential(
                nn.Conv2d(2 * dim, 2 * dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                # nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=bias)
            )

        self.fusion1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, bias=bias)
        self.fusion = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, bias=True)
        self.ffn = nn.Sequential(
                # nn.Conv2d(dim, dim, kernel_size=1, stride=1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias), nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
    def forward(self, input):
        x, y = input[0][0], input[0][1]
        x_K = [x,input[1]]
        '''
        #only global attention
        b, c, h, w = y.shape
        qkv = self.qkv_dwconv(self.layer_norm(y))
        q_dwconv, k_dwconv, v_dwconv = qkv.chunk(3, dim=1)
        Attention_qkv = self.Attention_qkv(x, q_dwconv, k_dwconv, v_dwconv)
        Attention_qkv = {0:Attention_qkv, 1:y}
        '''
        #global attention and local attention
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.layer_norm(y))
        q_dwconv, k_dwconv, v_dwconv = qkv.chunk(3, dim=1)
        Attention_qkv = self.Attention_qkv(x_K, q_dwconv, k_dwconv, v_dwconv)[0]

        degradation = self.degradation(self.layer_norm(y))
        degradation1 = degradation[:, c:, :, :]
        degradation2 = degradation[:, :c, :, :]
        input = self.input(self.layer_norm(x))
        input1 = input[:, c:, :, :]
        input2 = input[:, :c, :, :]
        ###############################
        main_kernel = self.main_kernel(torch.cat([input1, degradation1], dim=1))
        main_kernel = F.sigmoid(main_kernel)
        main_kernel_mul = torch.mul(main_kernel, degradation1)
        ###############################
        degradation_kernel = self.degradation_kernel(torch.cat([input2, degradation2], dim=1))
        degradation_kernel = F.sigmoid(degradation_kernel)
        degradation_kernel_mul = torch.mul(degradation_kernel, input2)
        out = self.fusion1(torch.cat([degradation_kernel_mul, main_kernel_mul], dim=1)) + x
        # out = degradation_kernel_mul + main_kernel_mul
        fusion1 = self.ffn(out) + out
        out = self.fusion(torch.cat([fusion1, Attention_qkv], dim=1)) + x
        out = [[out,y],x_K]
        return out

##########################################################################

class TMnet(nn.Module):
    def __init__(self, num_res=1, ResBlock=ResBlock_do_fft_bench,
        num_block = 1,
        num_blocks = [4,6,6,1,3], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        base_channel = 32,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        ):
        super(TMnet, self).__init__()
        self.num_blocks = num_blocks[0]
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res=4, ResBlock=ResBlock),
            EBlock(base_channel * 2, num_res=6, ResBlock=ResBlock),
            EBlock(base_channel * 4, num_res=6, ResBlock=ResBlock),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(1, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel, 1, kernel_size=3, relu=False, stride=1)
        ])

        self.decoder_level1=nn.Sequential(*[DPM(dim=base_channel, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)for i in range(num_blocks[4])]) 
        self.decoder_level2=nn.Sequential(*[DPM(dim=int(base_channel*2**1), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)for i in range(num_blocks[4])]) 
        self.decoder_level3=DBlock(base_channel * 4, num_res = 3, ResBlock=ResBlock)
        # self.decoder_level3=nn.Sequential(*[TransformerBlock(dim=int(base_channel*2**2), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.num_block = num_blocks[4]

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        ])

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 1, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 1, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.SAM = SAM(base_channel)
        self.SAM1 = SAM(base_channel * 2)
        self.SAM2 = SAM(base_channel * 4)

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)
        self.Refine = nn.ModuleList([
            FCN(base_channel * 4, base_channel * 4),
            FCN(base_channel * 2, base_channel * 2),
            FCN(base_channel, base_channel)
        ])

        self.AFF_fca1 = AFF_att_Fca(base_channel * 2, base_channel)

        self.AFF_fca2 = AFF_att_Fca(base_channel * 4, base_channel * 2)

    def forward(self, cloud):
        outputs = list()
        z2 = self.SCM2(cloud[1])
        z4 = self.SCM1(cloud[2])
        K1, _ = dynamic_prompt_K(cloud[0], 1)
        K2, num_modules_1 = dynamic_prompt_K(cloud[1], 2)
        K3, num_modules_2 = dynamic_prompt_K(cloud[2], 3)
        
        x1 = self.feat_extract[0](cloud[0])
        in1 = [x1, K1]
        res1 = self.Encoder[0](in1)[0]
        x2 = self.feat_extract[1](res1)
        z = self.FAM2(x2, z2)
        in2 = [z, K2]
        res2 = self.Encoder[1](in2)[0]

        x3 = self.feat_extract[2](res2)
        z = self.FAM1(x3, z4)
        in3 = [z, K3]
        z = self.Encoder[2](in3)[0]  # 128
        aff1 = self.AFFs[1](res1, res2, z)
        aff2 = self.AFFs[0](res1, res2, z)
        in4 = [z, K3]
        z = self.decoder_level3(in4)[0]
        sam2, rest2 = self.SAM2(z, cloud[2])
        # z_ = self.ConvsOut[0](z)
        # outputs.append(z_ + z4)
        outputs.append(rest2)
        z_up3 = self.feat_extract[3](z)  # 128-64 shangcaiyang

        # z = torch.cat([aff1, z], dim=1)
        z = self.AFF_fca2(aff1, z_up3)
        z = self.Convs[0](z)  # 128 - 64
        #z_pro1 = {0: z,1: z_up3}
        x_0 = {0: z, 1: z_up3}
        x_1 = {0: z, 1: z}
        x = x_0
        for i in range(self.num_block):
            if i <= num_modules_2:
                in5 = [x, K2]
                x = self.decoder_level2[i](in5)[0]
            else:
                x_1 = {0: x[0], 1: x[0]}
                in5 = [x_1, K2]
                x = self.decoder_level2[i](in5)[0]
        z = x[0]
        #z = self.Decoder[1](z_pro1,num_modules)[0]
        sam1, rest1 = self.SAM1(z, cloud[1])
        # z_ = self.ConvsOut[1](z)
        outputs.append(rest1)
        z_up2 = self.feat_extract[4](z)  # 64- 32
        # z = torch.cat([aff2, z], dim=1)
        z = self.AFF_fca1(aff2, z_up2)
        z = self.Convs[1](z)
        # z_pro2 = {0: z,1: z_up2}
        x_0 = {0: z, 1: z_up2}
        x_1 = {0: z, 1: z}
        x = x_0
        for i in range(self.num_block):
            if i <= num_modules_1:
                in6 = [x, K1]
                x = self.decoder_level1[i](in6)[0]
            else:
                x_1 = {0: x[0], 1: x[0]}
                in6 = [x_1, K1]
                x = self.decoder_level1[i](in6)[0]
        #z = self.Decoder[0](z_pro2,num_modules)[0]
        z = x[0]
        sam0, rest0 = self.SAM(z, cloud[0])  # 32
        # z = self.feat_extract[5](z)
        # outputs.append(z + cloud[0])
        outputs.append(rest0)

        re1 = self.Refine[0](sam2)  # 128
        out1 = self.feat_extract[5](re1)# + cloud[2]
        re2 = self.Refine[1](sam1)  # 64
        out2 = self.feat_extract[6](re2)# + cloud[1]
        re3 = self.Refine[2](sam0)  # 32
        out3 = self.feat_extract[7](re3)# + cloud[0]
        outputs.append(out1)
        outputs.append(out2)
        outputs.append(out3)

        return outputs

if __name__ == '__main__':
    model = TMnet()
    # import os
    # import pytorch_ssim
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #
    # # os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，忽略所有信息
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 忽略 warning 和 Error
    image1 = torch.randn((1, 1, 128, 128))
    image2 = torch.randn((1, 1, 128, 128))
    image3 = torch.randn((1, 1, 128, 128))
    image = [image1, image2, image3]
    model.to('cuda')
    
    # ssim_loss = pytorch_ssim.SSIM()
    start_time = time.time()
    with torch.no_grad():
        output1 = model([image1, image2, image3])
    print(output1[2].shape)
    print('training time:', time.time() - start_time)
#  print(ssim_loss(image1, image2).numpy())
