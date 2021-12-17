

from models import register_model
from models.transformer import *


@register_model('STAR-DCE-Sin')
class enhance_net_nopool(nn.Module):

    def __init__(self, number_f=24, depth=1, heads=8, dropout=0, patch_num=32):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 2, 1, bias=True)

        self.patch_num = patch_num
        self.transformer = Transformer(number_f, depth, heads, number_f, dropout)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=24, help="number of features")
        parser.add_argument("--depth", type=int, default=1, help="depth of transformer block")
        parser.add_argument("--heads", type=int, default=8, help="heads of transformer block")
        parser.add_argument("--dropout", type=float, default=0., help="dropout ratio of transformer block")
        parser.add_argument("--patch-num", type=int, default=32, help="number of patches")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f, depth=args.depth, heads=args.heads, dropout=args.dropout,
                   patch_num=args.patch_num)

    def forward(self, x, img_in=False):
        img_in = x if img_in is None else img_in
        n, c, h, w = x.shape
        x_ = F.interpolate(x, (256, 256))

        x1 = self.e_conv1(x_)
        x1 = self.relu(x1)
        x1 = nn.AdaptiveAvgPool2d((32, 32))(x1)
        trans_inp = rearrange(x1, 'b c  h w -> b (  h w)  c ')
        x_out = self.transformer(trans_inp)
        x_r = rearrange(x_out, 'b (h w) c -> b  c  h w', h=self.patch_num)
        x_r = F.upsample_bilinear(x_r, (h, w)).tanh()
        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')

        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)
        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        return enhanced_image, x_r_resize

@register_model('DCE-Net')
class enhance_net_dce_nopool(nn.Module):

    def __init__(self, number_f=32):
        super(enhance_net_dce_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=32, help="number of features")

    def forward(self, x, img_in=None):
        img_in = x if img_in is None else img_in

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)
        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)

        return enhanced_image, x_r_resize


@register_model('DCE-Net-Pool')
class enhance_net_dce_pool(nn.Module):

    def __init__(self, number_f):
        super(enhance_net_dce_pool, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)

        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=32, help="number of features")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f)

    def forward(self, x, img_in=None):
        img_in = x if img_in is None else img_in

        x1 = self.relu(self.e_conv1(x))
        x1 = self.maxpool(x1)
        x2 = self.relu(self.e_conv2(x1))
        x2 = self.maxpool(x2)
        x3 = self.relu(self.e_conv3(x2))
        x3 = self.maxpool(x3)
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x5 = self.upsample(x5)
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x6 = self.upsample(x6)
        x7 = self.e_conv7(torch.cat([x1, x6], 1))
        x7 = self.upsample(x7)

        x_r = F.tanh(x7)

        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)
        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)

        return enhanced_image, x_r_resize


@register_model('STAR-DCE-Ori')
class enhance_net_litr(nn.Module):

    def __init__(self, number_f=32, depth=1, heads=8, dropout=0, patch_num=32):
        super(enhance_net_litr, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f // 2, 2, 3, 1, 1, bias=True)

        self.patch_num = patch_num
        self.transformer = Transformer(number_f // 2, depth, heads, number_f // 2, dropout
                                       )
        self.pos_embedding = nn.Parameter(torch.randn(1, number_f // 2, 32, 32))

        self.enc_conv = nn.Conv2d(number_f // 2, number_f // 2, 3, 1, 1, bias=True)

        self.out_conv = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=32, help="number of features")
        parser.add_argument("--depth", type=int, default=1, help="depth of transformer block")
        parser.add_argument("--heads", type=int, default=8, help="heads of transformer block")
        parser.add_argument("--dropout", type=float, default=0., help="dropout ratio of transformer block")
        parser.add_argument("--patch-num", type=int, default=32, help="number of patches")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f, depth=args.depth, heads=args.heads, dropout=args.dropout,
                   patch_num=args.patch_num)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        img_in = x_in if img_in is None else img_in

        n, c, h, w = x_in.shape

        x1 = self.e_conv1(x_in)
        x1 = self.relu(x1)

        x_r_d = nn.AdaptiveAvgPool2d((h // 8, w // 8))(x1)
        x_conv = self.enc_conv(x_r_d)
        trans_inp = rearrange(x_r_d, 'b c  h w -> b (  h w)  c ')
        x_out = self.transformer(trans_inp)
        x_r = torch.cat([rearrange(x_out, 'b (h w) c -> b  c  h w', h=self.patch_num), x_conv], dim=1)
        x_r = self.out_conv(x_r)

        x_r = F.upsample_bilinear(x_r, (256, 256)).tanh()

        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)
        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        return enhanced_image, x_r_resize


@register_model('STAR-DCE-Half')
class enhance_net_litr_half(nn.Module):

    def __init__(self, number_f=16, depth=1, heads=8, dropout=0, patch_num=32):
        super(enhance_net_litr_half, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, number_f // 2, 2, 3, 1, 1, bias=True)

        self.patch_num = patch_num
        self.transformer = Transformer(number_f // 2, depth, heads, number_f // 2, dropout
                                       )
        self.pos_embedding = nn.Parameter(torch.randn(1, number_f // 2, 32, 32))

        self.enc_conv = nn.Conv2d(number_f // 2, number_f // 2, 3, 1, 1, bias=True)

        self.out_conv = nn.Conv2d(number_f, 24, 3, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--number-f", type=int, default=16, help="number of features")
        parser.add_argument("--depth", type=int, default=1, help="depth of transformer block")
        parser.add_argument("--heads", type=int, default=8, help="heads of transformer block")
        parser.add_argument("--dropout", type=float, default=0., help="dropout ratio of transformer block")
        parser.add_argument("--patch-num", type=int, default=32, help="number of patches")

    @classmethod
    def build_model(cls, args):
        return cls(number_f=args.number_f, depth=args.depth, heads=args.heads, dropout=args.dropout,
                   patch_num=args.patch_num)

    def forward(self, x_in, img_in=None):
        torch.autograd.set_detect_anomaly(True)
        img_in = x_in if img_in is None else img_in

        n, c, h, w = x_in.shape

        x1 = self.e_conv1(x_in)
        x1 = self.relu(x1)

        x_r_d = nn.AdaptiveAvgPool2d((h // 8, w // 8))(x1)
        x_conv = self.enc_conv(x_r_d)
        trans_inp = rearrange(x_r_d, 'b c  h w -> b (  h w)  c ')
        x_out = self.transformer(trans_inp)
        x_r = torch.cat([rearrange(x_out, 'b (h w) c -> b  c  h w', h=self.patch_num), x_conv], dim=1)
        x_r = self.out_conv(x_r)

        x_r = F.upsample_bilinear(x_r, (256, 256)).tanh()

        x_r_resize = F.interpolate(x_r, img_in.shape[2:], mode='bilinear')
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r_resize, 3, dim=1)
        x = img_in.to(x_r_resize.device)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhanced_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhanced_image_1 + r5 * (torch.pow(enhanced_image_1, 2) - enhanced_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhanced_image = x + r8 * (torch.pow(x, 2) - x)
        enhanced_image = torch.clamp(enhanced_image, 0, 1)

        return enhanced_image, x_r_resize

