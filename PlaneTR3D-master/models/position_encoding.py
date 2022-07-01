# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x,x_t,y_t):
        b, c, h, w = x.shape
        # print(b,c,h,w)
        # not_mask = torch.ones((b, h, w), dtype=torch.uint8, device=x.device)
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # x_embed = not_mask.cumsum(2, dtype=torch.float32)
        #修改
        x_embed=torch.ones(size=(1,b,h,w))*x_t
        y_embed=torch.ones(size=(1,b,h,w))*y_t

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        #修改
        #dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        # pos_x = x_embed[:, :, :, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        #修改
        pos_x = x_embed.permute(1,2,3,0)/ dim_t
        pos_y = y_embed.permute(1,2,3,0)/ dim_t

        #pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #修改
        pos_x = torch.cat((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=3).reshape(b,h,w,self.num_pos_feats)
        pos_y = torch.cat((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=3).reshape(b,h,w,self.num_pos_feats)


        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def build_position_encoding(position_embedding_mode='sine', hidden_dim=256):
    N_steps = hidden_dim // 2
    # if position_embedding_mode in ('v2', 'sine'):
    #     # TODO find a better way of exposing other arguments
    #     position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    # else:
    #     raise ValueError(f"not supported {position_embedding_mode}")
    position_embedding = PositionEmbeddingSine(N_steps, normalize=True)

    return position_embedding
