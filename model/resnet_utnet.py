import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_utils import up_block
from .transunet import ResNetV2
from .conv_trans_utils import block_trans
import pdb




class ResNet_UTNet(nn.Module):
    def __init__(self, in_ch, num_class, reduce_size=8, block_list='234', num_blocks=[1,2,4], projection='interp', num_heads=[4,4,4], attn_drop=0., proj_drop=0., rel_pos=True, block_units=(3,4,9), width_factor=1):
        
        super().__init__()
        self.resnet = ResNetV2(block_units, width_factor)


        if '0' in block_list:
            self.trans_0 = block_trans(64, num_blocks[-4], 64//num_heads[-4], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)
        else:
            self.trans_0 = nn.Identity()


        if '1' in block_list:
            self.trans_1 = block_trans(256, num_blocks[-3], 256//num_heads[-3], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.trans_1 = nn.Identity()

        if '2' in block_list:
            self.trans_2 =  block_trans(512, num_blocks[-2], 512//num_heads[-2], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.trans_2 = nn.Identity()

        if '3' in block_list:
            self.trans_3 = block_trans(1024, num_blocks[-1], 1024//num_heads[-1], attn_drop=attn_drop, proj_drop=proj_drop, reduce_size=reduce_size, projection=projection, rel_pos=rel_pos)

        else:
            self.trans_3 = nn.Identity()
        
        self.up1 = up_block(1024, 512, scale=(2,2), num_block=1)
        self.up2 = up_block(512, 256, scale=(2,2), num_block=1)
        self.up3 = up_block(256, 64, scale=(2,2), num_block=1)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.output = nn.Conv2d(64, num_class, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        if x.shape[1] == 1: 
            x = x.repeat(1, 3, 1, 1)
        x, features = self.resnet(x)

        out3 = self.trans_3(x)
        out2 = self.trans_2(features[0])
        out1 = self.trans_1(features[1])
        out0 = self.trans_0(features[2])

        out = self.up1(out3, out2)
        out = self.up2(out, out1)
        out = self.up3(out, out0)
        out = self.up4(out)

        out = self.output(out)

        return out
            


