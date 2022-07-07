import torch
from torch import nn
from mmcv.ops.point_sample import bilinear_grid_sample#1.3.8+
from models.position_encoding import build_position_encoding
from models.transformer import build_transformer
from models.HRNet import build_hrnet
import os
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as tf
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')


class My_HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(My_HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        """检查分支数是否一致"""
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=0.01),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i],
                                       momentum=0.01),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=0.01)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=0.01),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            # print(self.fuse_layers)
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    # print("y",y.shape)
                    # print(self.fuse_layers[i][j](x[j]).shape)
                    # print("x",x[j].shape)
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def my_build_hrnet():
    backbone=My_HighResolutionNet()#HighResolutionNet
    return backbone

class My_HighResolutionNet(nn.Module):
    def __init__(self):

        super(My_HighResolutionNet, self).__init__()

        #stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01,track_running_stats=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.ReLU()


        num_channels_1 = 64
        block_1 = Bottleneck
        num_blocks_1 = 4
        self.layer1 = self._make_layer(block_1, 64, num_channels_1, num_blocks_1)
        stage1_out_channel = block_1.expansion * num_channels_1  # 4*64

        num_channels_2 = [32,64]
        block_2 = BasicBlock
        NUM_MODULES_2=1
        NUM_BRANCHES_2=2
        NUM_BLOCKS_2=[4,4]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels_2)
        self.stage2, pre_stage_channels = self._make_stage(
            NUM_MODULES_2,NUM_BRANCHES_2,NUM_BLOCKS_2,block_2, num_channels_2)

        num_channels_3 = [32,64,128]
        block_3 = BasicBlock
        NUM_MODULES_3=4
        NUM_BRANCHES_3=3
        NUM_BLOCKS_3=[4,4,4]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels_3)
        self.stage3, pre_stage_channels = self._make_stage(
            NUM_MODULES_3,NUM_BRANCHES_3,NUM_BLOCKS_3,block_3, num_channels_3)

        num_channels_4 = [32,64,128,256]
        block_4 = BasicBlock
        NUM_MODULES_4=3
        NUM_BRANCHES_4=4
        NUM_BLOCKS_4=[4,4,4,4]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels_4)
        self.stage4, pre_stage_channels = self._make_stage(
            NUM_MODULES_4,NUM_BRANCHES_4,NUM_BLOCKS_4,block_4, num_channels_4, multi_scale_output=True)

        self.init_weights('ckpts/hrnetv2_w32_imagenet_pretrained.pth')

        self.out_channels = num_channels_4

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 0<1 第一个
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=0.01),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=0.01),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self,NUM_MODULES,NUM_BRANCHES,NUM_BLOCKS,block, num_inchannels,
                    multi_scale_output=True):
        num_modules = NUM_MODULES
        num_branches = NUM_BRANCHES
        num_blocks = NUM_BLOCKS
        num_channels =num_inchannels
        fuse_method = 'SUM'

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                My_HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.layer1(x)
        #
        # x_list = []
        # for i in range(2):
        #     # if self.transition1[i] is not None:
        #     x_list.append(self.transition1[i](x))
        #     # else:
        #     #     x_list.append(x)
        # y_list = self.stage2(x_list)
        #
        # x_list = []
        # for i in range(3):
        #     if self.transition2[i] is not None:
        #         x_list.append(self.transition2[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage3(x_list)
        #
        # x_list = []
        # for i in range(4):
        #     if self.transition3[i] is not None:
        #         x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # y_list = self.stage4(x_list)
        #
        # return y_list
        return x

    def init_weights(self, pretrained='', ):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            # print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}

            assert len(pretrained_dict.keys()) > 50
            # for k, _ in pretrained_dict.items():
            #     logger.info(
            #         '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print("error: can not find %s"%(pretrained))
            exit()

#Transformer:
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def get_lines_features(feat, pos, lines, size_ori, n_pts=21):
    """
    :param feat: B, C, H, W
    :param lines: B, N, 4
    :return: B, C, N
    """
    ho, wo = size_ori
    b, c, hf, wf = feat.shape
    line_num = lines.shape[1]

    with torch.no_grad():
        scale_h = ho / hf
        scale_w = wo / wf
        scaled_lines = lines.clone()
        # print("scaled_lines:",scaled_lines.shape)#1,200,4
        scaled_lines[:, :, 0] = scaled_lines[:, :, 0] / scale_w
        scaled_lines[:, :, 1] = scaled_lines[:, :, 1] / scale_h
        scaled_lines[:, :, 2] = scaled_lines[:, :, 2] / scale_w
        scaled_lines[:, :, 3] = scaled_lines[:, :, 3] / scale_h

        spts, epts = torch.split(scaled_lines, (2, 2), dim=-1)  # B, N, 2

        if n_pts > 2:
            delta_pts = (epts - spts) / (n_pts-1)  # B, N, 2
            delta_pts = delta_pts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            steps = torch.linspace(0., n_pts-1, n_pts).view(1, 1, n_pts, 1).to(device=lines.device)

            spts_expand = spts.unsqueeze(dim=2).expand(b, line_num, n_pts, 2)  # b, n, n_pts, 2
            line_pts = spts_expand + delta_pts * steps  # b, n, n_pts, 2

        elif n_pts == 2:
            line_pts = torch.stack([spts, epts], dim=2)  # b, n, n_pts, 2
        elif n_pts == 1:
            line_pts = torch.cat((spts, epts), dim=1).unsqueeze(dim=2)

        line_pts[:, :, :, 0] = line_pts[:, :, :, 0] / (wf-1) * 2. - 1.
        line_pts[:, :, :, 1] = line_pts[:, :, :, 1] / (hf-1) * 2. - 1.

        line_pts = line_pts.detach()

    sample_feats=bilinear_grid_sample(feat,line_pts)
    # sample_feats = F.grid_sample(feat, line_pts)  # b, c, n, n_pts

    b, c, ln, pn = sample_feats.shape
    sample_feats = sample_feats.permute(0, 1, 3, 2).contiguous().view(b, -1, ln)

    sample_pos=bilinear_grid_sample(pos,line_pts)
    # sample_pos = F.grid_sample(pos, line_pts)
    sample_pos = torch.mean(sample_pos, dim=-1)

    return sample_feats, sample_pos  # b, c, n

class top_down(nn.Module):
    def __init__(self, in_channels=[], channel=64, m_dim=256, double_upsample=False):
        super(top_down, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.double_upsample = double_upsample

        # top down
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if double_upsample:
            self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.up_conv3 = conv_bn_relu(channel, channel, 1)
        self.up_conv2 = conv_bn_relu(channel, channel, 1)
        self.up_conv1 = conv_bn_relu(channel, channel, 1)

        # lateral
        self.c4_conv = conv_bn_relu(in_channels[3], channel, 1)
        self.c3_conv = conv_bn_relu(in_channels[2], channel, 1)
        self.c2_conv = conv_bn_relu(in_channels[1], channel, 1)
        self.c1_conv = conv_bn_relu(in_channels[0], channel, 1)

        self.m_conv_dict = nn.ModuleDict({})
        self.m_conv_dict['m4'] = conv_bn_relu(m_dim, channel)

        self.p0_conv = nn.Conv2d(channel, channel, (3, 3), padding=1)

    def init_weights(self):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, memory):
        c1, c2, c3, c4 = x
        # print("c4",c4.shape) #1,256,8,12
        # print("memory",memory.shape)#1.256.8.12
        p4 = self.c4_conv(c4) + self.m_conv_dict['m4'](memory)
        # print(p4.shape)# 1,64,8,12
        p3 = self.up_conv3(self.upsample(p4)) + self.c3_conv(c3)

        p2 = self.up_conv2(self.upsample(p3)) + self.c2_conv(c2)

        p1 = self.up_conv1(self.upsample(p2)) + self.c1_conv(c1)

        p0 = self.upsample(p1)

        p0 = self.relu(self.p0_conv(p0))

        return p0, p1, p2, p3, p4
        # return p4

def conv_bn_relu(in_dim, out_dim, k=1, pad=0):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, (k, k), padding=pad, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(inplace=True)
    )

#PlaneTR_HRNet
class PlaneModel(torch.nn.Module):
    def __init__(self,position_embedding_mode='sine'):
        super(PlaneModel, self).__init__()
        num_queries=20
        plane_embedding_dim = 8
        loss_layer_num = 1
        predict_center = True
        use_lines = False

        # Feature extractor
        self.backbone = my_build_hrnet()#build_hrnet() HighResolutionNet
        self.backbone_channels = self.backbone.out_channels
        #tinynet.onnx ↑

        #pre-defined
        self.loss_layer_num = loss_layer_num
        assert self.loss_layer_num <= 2
        self.return_inter = False
        self.predict_center = predict_center
        self.use_lines = use_lines
        self.num_sample_pts = 2
        self.if_predict_depth = True
        self.if_shareHeads = True

        self.hidden_dim = 256
        self.num_queries = num_queries
        self.context_channels = self.backbone_channels[-1]#256
        self.line_channels = self.backbone_channels[1]

        self.lines_reduce = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_sample_pts, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim), )

        self.plane_embedding_dim = plane_embedding_dim
        self.channel = 64

        # Transformer Branch
        self.input_proj = nn.Conv2d(self.context_channels, self.hidden_dim, kernel_size=1)

        self.lines_proj = nn.Conv2d(self.line_channels, self.hidden_dim, kernel_size=1)
        self.position_embedding = build_position_encoding(position_embedding_mode, hidden_dim=self.hidden_dim)
        self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)
        self.transformer = build_transformer(hidden_dim=self.hidden_dim, dropout=0.1, nheads=8, dim_feedforward=1024,
                                             enc_layers=6, dec_layers=6, pre_norm=True,
                                             return_inter=self.return_inter,
                                             use_lines=use_lines, loss_layer_num=self.loss_layer_num)
        # # instance-level plane embedding
        self.plane_embedding = MLP(self.hidden_dim, self.hidden_dim, self.plane_embedding_dim, 3)
        # plane / non-plane classifier
        self.plane_prob = nn.Linear(self.hidden_dim, 2 + 1)
        # instance-level plane 3D parameters
        self.plane_param = MLP(self.hidden_dim, self.hidden_dim, 3, 3)
        # instance-level plane center
        self.plane_center = MLP(self.hidden_dim, self.hidden_dim, 2, 3)

        # Convolution Branch
        # top_down
        self.top_down = top_down(self.backbone_channels, self.channel, m_dim=self.hidden_dim, double_upsample=False)
        # pixel embedding
        self.pixel_embedding = nn.Conv2d(self.channel, self.plane_embedding_dim, (1, 1), padding=0)
        # pixel-level plane center
        self.pixel_plane_center = nn.Conv2d(self.channel, 2, (1, 1), padding=0)
        # pixel-level depth

        self.top_down_depth = top_down(self.backbone_channels, self.channel, m_dim=self.hidden_dim,
                                       double_upsample=False)
        self.depth = nn.Conv2d(self.channel, 1, (1, 1), padding=0)

    #思路是，先实现每个小的模块，再集成
    def forward(self,x,x_t,y_t):
        #添加正则化
        transforms = tf.Compose([
            # tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x=transforms(x[0]).unsqueeze(0)

        bs, _, ho, wo = x.shape
        x=self.backbone(x)

        output={'bnx':x}
        return output


        # c1, c2, c3, c4 = self.backbone(x)  # c: 32, 64, 128, 256
        #
        # # context src feature
        # src = c4
        #
        # # context feature proj
        # src = self.input_proj(src).to(device)  # b, hidden_dim, h, w ok
        # # position embedding
        # pos = self.position_embedding(src,x_t,y_t)#1 256 2 2  1 2 2 256
        #
        # hs_all, _, memory = self.transformer(src, self.query_embed.weight, pos, tgt=None,
        #                                      src_lines=None, mask_lines=None,
        #                                      pos_embed_lines=None)  # memory: b, c, h, w
        # # return x,src,pos
        # hs = hs_all[-self.loss_layer_num:, :, :, :].contiguous()  # dec_layers, b, num_queries, hidden_dim
        # # plane embedding
        # plane_embedding = self.plane_embedding(hs)  # dec_layers, b, num_queries, 2
        #
        # # plane classifier
        # plane_prob = self.plane_prob(hs)  # dec_layers, b, num_queries, 3
        # # plane parameters
        # plane_param = self.plane_param(hs)  # dec_layers, b, num_queries, 3
        # # plane center
        # plane_center = self.plane_center(hs)  # dec_layers, b, num_queries, 2
        # plane_center = torch.sigmoid(plane_center)
        # p0, p1, p2, p3, p4 = self.top_down((c1, c2, c3, c4), memory)
        #
        # pixel_embedding = self.pixel_embedding(p0)  # b, 2, h, w 8
        # pixel_center = self.pixel_plane_center(p0)  # b, 2, h, w 2
        #
        # pixel_center = torch.sigmoid(pixel_center)  # 0~1
        # p_depth, _, _, _, _ = self.top_down_depth((c1, c2, c3, c4), memory)
        # pixel_depth = self.depth(p_depth)
        #
        #
        # output = {'pred_logits': plane_prob[-1], 'pred_param': plane_param[-1],
        #           'pred_plane_embedding': plane_embedding[-1], 'pixel_embedding': pixel_embedding}
        # output['pixel_center'] = pixel_center
        # output['pred_center'] = plane_center[-1]
        # output['pixel_depth'] = pixel_depth
        # output['x']=x
        # output['src']=src
        # return output

        # output=[]
        # output.append(plane_prob[-1])
        # output.append(plane_param[-1])
        # output.append(plane_embedding[-1])
        # output.append(pixel_embedding)
        # output.append(pixel_center)
        # output.append(plane_center[-1])
        # output.append(pixel_depth)
        #
        # num_queries=20
        # embedding_dist_threshold=1.0

        # pred_logits = plane_prob[-1]
        # pred_plane_embedding = plane_embedding
        # pred_pixel_embedding = pixel_embedding

        # pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
        # score, labels = pred_prob.max(dim=-1)
        # # label_mask = labels > 0
        # label_mask=torch.where(labels>0,1,0)
        # if sum(label_mask) == 0:
        #     _, max_pro_idx = pred_prob[:, 1].max(dim=0)
        #     label_mask[max_pro_idx] = 1
        #
        # valid_plane_embedding = pred_plane_embedding[:,:,label_mask[0], :]#1 1 20 8 1 20 8 1
        #
        # c_embedding = pred_plane_embedding.shape[-1]#2
        # flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
        # dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)# 1 1  1024 20
        # #修改
        # flat_pixel_embedding = pred_pixel_embedding.view(1,c_embedding,1, -1).repeat(1,1,20,1).permute(0,3,2,1)#1 hw,1, c_embedding
        # # U 1,20,8,1024 ↑ ↓
        # valid_plane_embedding=valid_plane_embedding.repeat(1,ho*wo,1,1)
        #TODO unity处理tensor数据; 或者解决以下代码转成onnx在unity运行 目前似乎两个方法都不行

        # dist_map_pixel2planes=torch.sub(flat_pixel_embedding,valid_plane_embedding)
        # dist_map_pixel2planes=torch.sqrt(torch.sum((flat_pixel_embedding-valid_plane_embedding)**2,3)).unsqueeze(0)
        # # dist_map_pixel2planes=torch.sum(torch.abs(torch.sub(flat_pixel_embedding,valid_plane_embedding)),3).unsqueeze(0)
        # dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
        # dist_pixel2onePlane = dist_pixel2onePlane.view(ho, wo)  # h, w
        # planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(ho, wo)  # h, w
        # mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w
        # predict_segmentation = planeIdx_pixel2onePlane #.cpu().numpy().copy()  # h, w
        #
        # if int(mask_pixelOnPlane.sum()) < (ho * wo):  # set plane idx of non-plane pixels as num_queries + 1
        #     predict_segmentation[mask_pixelOnPlane.cpu() == 0] = num_queries + 1
        # predict_segmentation = predict_segmentation.reshape(ho,
        #                                                     wo)  # h, w (0~num_queries-1:plane idx, num_queries+1:non-plane)
        # predict_segmentation[predict_segmentation == (num_queries + 1)] = -1
        #
        # segmentation = predict_segmentation  # -1 indicates non-plane region
        # assert segmentation.dim() == 3 or segmentation.dim() == 2
        # if segmentation.dim() == 3:
        #     assert segmentation.shape[0] == 1
        #     segmentation = segmentation[0]#.cpu().numpy()
        # else:
        #     segmentation = segmentation#.cpu().numpy()
        # # print(np.unique(segmentation))
        # segmentation += 1
        # colors = torch.from_numpy(labelcolormap(256))
        # # ***************  get color segmentation
        # seg = torch.stack([colors[segmentation, 0], colors[segmentation, 1], colors[segmentation, 2]], axis=2)
        # # ***************  get blend image
        # x=x.squeeze(0).permute(1,2,0)
        # blend_seg = (seg * 0.7 + x* 0.3)
        # seg_mask = (segmentation > 0)
        # seg_mask = seg_mask[:, :, ].unsqueeze(2)
        # blend_seg = blend_seg * seg_mask + x * (1-seg_mask)
        #
        # return blend_seg


        # if self.loss_layer_num > 1 and self.training:
        #     assert plane_prob.shape[0] == 2
        #     # import pdb; pdb.set_trace()
        #     aux_outputs = []
        #     aux_l = {'pred_logits': plane_prob[0], 'pred_plane_embedding': plane_embedding[0],
        #              'pixel_embedding': pixel_embedding}
        #     if self.predict_center:
        #         aux_l['pred_center'] = plane_center[0]
        #     aux_outputs.append(aux_l)
        #
        # output['aux_outputs'] = aux_outputs
        # return pos
        # return output


checkpoint = './ckpts/PlaneTR_Pretrained.pt'
onnx_path = './PlaneTR_test.onnx'

input_0 = torch.randn(1, 3, 32,32)#b 3 h w
input_1=torch.linspace(1,2,2).unsqueeze(1)
input_2=torch.linspace(1,2,2).unsqueeze(0)
# input_1=torch.randn(1,200,4)#lines
# input_2=torch.randn(1)#num_lines
# input_names = ['input_0', 'input_1', 'input_2']
input_names = ['input_0','input_1','input_2']
# output_names=['output_0','output_1']
# output_names=['output_0','output_1','output_2']
output_names=['output_0']

weight=torch.load(checkpoint,map_location=device)
weight2={}
for k in weight.keys():
    if k=="backbone.bn1.running_mean" or k=="backbone.bn1.running_var" or k =="backbone.bn1.num_batches_tracked":
        pass
    else:
        weight2[k]=weight[k]

model = PlaneModel()  # 导入模型
model.eval()
model.to(device)
model.load_state_dict(weight2)# 初始化权重


input=(input_0.to(device),input_1.to(device),input_2.to(device))
torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                  output_names=output_names,dynamic_axes={'input_0' : {0 :"batch",2:"height",3:"weight"} ,
                                                          'input_1' : {0 :"height"},
                                                          'input_2' : {1:"weight"}
                                                          # variable lenght axes
},opset_version=11)  # 指定模型的输入，以及onnx的输出路径

