import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv

import functools


class DP_conv(nn.Module):  # dense pointwise convolution
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super(DP_conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_channel,   
            out_channels=in_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel
        )  # depthwise(DW)conv
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        ) # pointwise(PW)conv

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out


class deformable_SKConv(nn.Module):
    def __init__(self, in_fea, out_fea, in_nc=7, branches=3, reduce=16, len=32):
        super(deformable_SKConv, self).__init__()
        self.in_nc = in_nc
        self.branches = branches

        len = max(in_fea // reduce, len)
        self.offset_mask = nn.ModuleList([])
        self.deform_conv = nn.ModuleList([])
        for i in range(branches):
            d_size = (2 * i + 1) ** 2
            self.offset_mask.append(DP_conv(in_channel=in_fea, out_channel=in_nc * 3 * d_size, kernel_size=2 * i + 1, stride=1))
            self.deform_conv.append(ModulatedDeformConv(in_nc, out_fea, kernel_size=2 * i + 1, stride=1, padding=(2 * i + 1) // 2, deformable_groups=in_nc))

        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels=out_fea * branches, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(out_fea, len, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, out_fea, kernel_size=1, stride=1)
            )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_fea * branches, out_channels=out_fea, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, fea, inputs):
        out = []
        for i in range(self.branches):
            d_size = (2 * i + 1) ** 2
            offset_mask = self.offset_mask[i](fea)
            offset = offset_mask[:, :self.in_nc * 2 * d_size, ...]
            mask = torch.sigmoid(offset_mask[:, self.in_nc * 2 * d_size:, ...])
            fused_feat = F.relu(self.deform_conv[i](inputs, offset, mask), inplace=True)
            out.append(fused_feat)

        out = torch.stack(out, dim=1)
        b, t, c, h, w = out.shape
        attention = out.view(b, -1, h, w)
        attention = self.conv_attention(attention)

        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        # print(attention.shape)
        out = out * attention  # b, 3, c, h, w

        out = out.view(b, -1, h, w)
        out = self.conv(out)
        return out

# ==========
# Spatio-temporal deformable fusion module
# ==========
class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF, self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                    )
                )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                    )
                )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
            )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
            )

        self.d_SKConv = deformable_SKConv(in_fea=nf, out_fea=out_nc, in_nc=in_nc, branches=3, reduce=4, len=32)

        # self.offset_mask = nn.Conv2d(nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2)  
        # self.deform_conv = ModulatedDeformConv(in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc)  

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        # print(out_lst[0].shape)
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
                )

        # compute offset and mask
        out = self.out_conv(out)
        out = self.d_SKConv(out, inputs)
        return out

        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
            )
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
            )
        return fused_feat


class CA_block(nn.Module):   
    def __init__(self, in_channel=32, reduce_ratio=4):
        super(CA_block, self).__init__()
        self.ca_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduce_ratio, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channel // reduce_ratio, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.ca_layer(x)
        x = x * x1
        return x


class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(dense_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growthRate, kernel_size=3, stride=1, padding=1)   
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


class Ada_RDBlock(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, reduce_ratio=4, a=1, b=0.2):
        super(Ada_RDBlock, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.ca_block = CA_block(in_channel=in_channels_, reduce_ratio=reduce_ratio)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels_, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(a)
        self.fuse_weight_1.data.fill_(b)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.ca_block(out)
        out = self.conv3x3(out)
        return x * self.fuse_weight + out * self.fuse_weight_1


# ==========
# Network
# ==========
class Net(nn.Module):
    """STDF -> QE -> residual.
    
    in: (B T*C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        super(Net, self).__init__()
        self.radius = opts_dict['radius']
        self.input_len = 2 * self.radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']

        self.ffnet = STDF(in_nc= self.in_nc * self.input_len, out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], nb=opts_dict['stdf']['nb'], deform_ks=opts_dict['stdf']['deform_ks'])
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=opts_dict['stdf']['out_nc'], out_channels=opts_dict['Ada_RDBlock']['in_nc'], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.Ada_RDBlock_layer = self.make_layer(functools.partial(Ada_RDBlock, 
                             opts_dict['Ada_RDBlock']['in_nc'], opts_dict['Ada_RDBlock']['growthRate'], opts_dict['Ada_RDBlock']['num_layer'], 
                             opts_dict['Ada_RDBlock']['reduce_ratio'], opts_dict['Ada_RDBlock']['a'], opts_dict['Ada_RDBlock']['b']), 
                                                                  opts_dict['Ada_RDBlock_num']) 

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=opts_dict['Ada_RDBlock']['in_nc'], out_channels=opts_dict['Ada_RDBlock']['in_nc'], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv_last = nn.Conv2d(in_channels=opts_dict['Ada_RDBlock']['in_nc'], out_channels=opts_dict['stdf']['in_nc'], kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.ffnet(x)
        out = self.conv1(out)  # 64

        out = self.Ada_RDBlock_layer(out)
        out = self.conv3(out)
        out = self.conv_last(out)
        
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)]
        out += x[:, frm_lst, ...]  # res: add middle frame
        return out

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)




if __name__ == "__main__":
    import argparse
    import yaml
    import numpy as np
    import os.path as op
    def receive_arg():
        """Process all hyper-parameters and experiment settings.
        Record in opts_dict."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--opt_path', type=str, default='option_mfqev2_1G.yml', help='Path to option YAML file.')
        parser.add_argument('--local_rank', type=int, default=0, help='Distributed launcher requires.')  
        args = parser.parse_args()
        with open(args.opt_path, 'r') as fp:
            opts_dict = yaml.load(fp, Loader=yaml.FullLoader)
        opts_dict['opt_path'] = args.opt_path
        opts_dict['train']['rank'] = args.local_rank
        if opts_dict['train']['exp_name'] == None:
            opts_dict['train']['exp_name'] = utils.get_timestr()
        opts_dict['train']['log_path'] = op.join("exp", opts_dict['train']['exp_name'], "log.log")
        opts_dict['train']['checkpoint_save_path_pre'] = op.join("exp", opts_dict['train']['exp_name'], "ckp_")
        
        opts_dict['train']['num_gpu'] = torch.cuda.device_count()   # Returns the number of GPUs available.
        if opts_dict['train']['num_gpu'] > 1:
            opts_dict['train']['is_dist'] = True
        else:
            opts_dict['train']['is_dist'] = False
        opts_dict['test']['restore_iter'] = int(opts_dict['test']['restore_iter'])  # 290000
        return opts_dict

    opts_dict = receive_arg()

    x = torch.rand(1, 7, 96, 96).cuda(1)
    model = Net(opts_dict=opts_dict['network']).cuda(1) 
    # model(x)
    print(model(x).shape)

    from thop import profile
    flops, params = profile(model, inputs=(x, ))
    print(flops, params)  # 5209307840.0 1254200.0


    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(1.0 * params)  # 2219550.0        2205600.0

