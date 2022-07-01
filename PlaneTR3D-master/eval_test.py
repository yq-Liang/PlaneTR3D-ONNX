import scipy.io as sio
import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
import yaml
import sys

import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision.transforms as tf
transforms = tf.Compose([
        tf.ToTensor(),
        # tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

from change_utils.utils import Set_Config, Set_Logger, Set_Ckpt_Code_Debug_Dir

from models.planeTR_HRNet import PlaneTR_HRNet as PlaneTR
from models.ScanNetV1_PlaneDataset import scannetv1_PlaneDataset

from change_utils.misc import AverageMeter, get_optimizer, get_coordinate_map

from change_utils.metric import eval_plane_recall_depth, eval_plane_recall_normal, evaluateMasks

from change_utils.disp import plot_depth_recall_curve, plot_normal_recall_curve, visualizationBatch

from Test_toONNX import PlaneModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
import argparse


def eval():
    # set random seed
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network
    network = PlaneModel()
    # network.eval()
    # load nets into gpu or cpu
    network = network.to(device)
    network.load_state_dict(torch.load('./ckpts/PlaneTR_Pretrained.pt', map_location=torch.device('cpu')))


    k_inv_dot_xy1 = get_coordinate_map(device)
    num_queries = 20
    embedding_dist_threshold = 1.0

    # define metrics
    pixelDepth_recall_curve = np.zeros((13))
    planeDepth_recall_curve = np.zeros((13, 3))
    pixelNorm_recall_curve = np.zeros((13))
    planeNorm_recall_curve = np.zeros((13, 3))
    plane_Seg_Metric = np.zeros((3))

    data_path = './res/test.png'
    # data_path = './res/test32.png'

    image = cv2.imread(data_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 192 256 3

    input = image.astype(np.float32)
    input = torch.tensor(input)# h w 3
    input = input.unsqueeze(1)
    # print(input.shape)
    input = input.permute(1, 3, 0, 2)
    # print(input.shape)
    image= input

    ##下面这个读取方法不行
    # image = Image.fromarray(image)
    # image = transforms(image)
    # image = image.unsqueeze(0)
    # input = image.numpy()

    bs, _, h, w = image.shape
    # assert bs == 1, "batch size should be 1 when testing!"
    # assert h == 192 and w == 256
    x_t=torch.linspace(1,h//16,h//16).unsqueeze(1)
    y_t=torch.linspace(1,w//16,w//16).unsqueeze(0)
    # forward pass
    outputs = network(image,x_t,y_t)
    # outputs = network(image)
    # cv2.imwrite('test.png', outputs)

    # decompose outputs
    pred_logits = outputs['pred_logits'][0]  # num_queries, 3
    pred_param = outputs['pred_param'][0]  # num_queries, 3
    pred_plane_embedding = outputs['pred_plane_embedding'][0]  # num_queries, 2
    pred_pixel_embedding = outputs['pixel_embedding'][0]  # 2, h, w

    pred_pixel_depth = outputs['pixel_depth'][0, 0]  # h, w

    # #
    # # print(2, pred_param.shape)
    # # print(3, pred_plane_embedding.shape)
    # # print(4, pred_pixel_embedding.shape)
    # # print(5, pred_pixel_depth.shape)
    # #log file
    # print(outputs.shape)
    path=r'res/eval_test_file.txt'
    file=open(path,'w+')
    for i in outputs.keys():
        # print(outputs[i].shape)
        file.write(str(i))
        file.write(":\n")
        file.write(str(outputs[i]))
        file.write("\n")
    file.close()


    # remove non-plane instance
    pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
    print("pred_prob",pred_prob.shape)
    score, labels = pred_prob.max(dim=-1)
    labels[labels != 1] = 0
    label_mask = labels > 0
    print("label_mask",label_mask.shape)
    if sum(label_mask) == 0:
        _, max_pro_idx = pred_prob[:, 1].max(dim=0)
        label_mask[max_pro_idx] = 1
    valid_param = pred_param[label_mask, :]  # valid_plane_num, 3
    valid_plane_embedding = pred_plane_embedding[label_mask, :]  # valid_plane_num, c_embedding
    valid_plane_num = valid_plane_embedding.shape[0]
    valid_plane_prob = score[label_mask]  # valid_plane_num
    assert valid_plane_num <= num_queries

    # calculate dist map
    c_embedding = pred_plane_embedding.shape[-1]
    flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
    dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)  # hw, valid_plane_num
    dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
    dist_pixel2onePlane = dist_pixel2onePlane.view(h, w)  # h, w
    planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(h, w)  # h, w
    mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w

    # get plane segmentation
    # gt_seg = gt_seg.reshape(h, w)  # h, w
    predict_segmentation = planeIdx_pixel2onePlane.cpu().numpy().copy()  # h, w
    if int(mask_pixelOnPlane.sum()) < (h * w):  # set plane idx of non-plane pixels as num_queries + 1
        predict_segmentation[mask_pixelOnPlane.cpu().numpy() == 0] = num_queries + 1
    predict_segmentation = predict_segmentation.reshape(h, w)  # h, w (0~num_queries-1:plane idx, num_queries+1:non-plane)

    # get depth map
    depth_maps_inv = torch.matmul(valid_param, k_inv_dot_xy1)
    depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
    depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)
    inferred_depth = depth_maps.t()[range(h * w), planeIdx_pixel2onePlane.view(-1)].view(h, w)
    inferred_depth = inferred_depth * mask_pixelOnPlane.float() + pred_pixel_depth * (1-mask_pixelOnPlane.float())

    # get depth maps
    # gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)  # h, w
    inferred_depth = inferred_depth.detach().numpy().reshape(h, w)
    inferred_depth = np.clip(inferred_depth, a_min=1e-4, a_max=10.)
    #
    # # # ----------------------------------------------------- evaluation
    # # # 1 evaluation: plane segmentation
    # # pixelStatistics, planeStatistics = eval_plane_recall_depth(
    # #     predict_segmentation, gt_seg, inferred_depth, gt_depth, valid_plane_num)
    # # pixelDepth_recall_curve += np.array(pixelStatistics)
    # # planeDepth_recall_curve += np.array(planeStatistics)
    # #
    # # # 2 evaluation: plane segmentation
    # # instance_param = valid_param.cpu().numpy()
    # # gt_plane_instance_parameter = gt_plane_instance_parameter.cpu().numpy()
    # # plane_recall, pixel_recall = eval_plane_recall_normal(predict_segmentation, gt_seg,
    # #                                                                 instance_param, gt_plane_instance_parameter,
    # #                                                                 pred_non_plane_idx=num_queries+1)
    # # pixelNorm_recall_curve += pixel_recall
    # # planeNorm_recall_curve += plane_recall
    # #
    # # # 3 evaluation: plane segmentation
    # # plane_Seg_Statistics = evaluateMasks(predict_segmentation, gt_seg, device, pred_non_plane_idx=num_queries+1)
    # # plane_Seg_Metric += np.array(plane_Seg_Statistics)
    # #
    # # # ------------------------------------ log info
    # # print(f"RI(+):{plane_Seg_Statistics[0]:.3f} | VI(-):{plane_Seg_Statistics[1]:.3f} | SC(+):{plane_Seg_Statistics[2]:.3f}")
    # #
    # # # ---------------------------------- debug: visualization
    # #
    # # # GT
    # # gt_seg[gt_seg == 20] = -1
    # # gt_seg = torch.from_numpy(gt_seg)
    # # gt_depth = torch.from_numpy(gt_depth)
    # # debug_dict = {'image': image[0].detach(), 'segmentation': gt_seg.detach(),
    # #               'depth_GT': gt_depth.detach(),
    # #               'K_inv_dot_xy_1': k_inv_dot_xy1,}
    # # visualizationBatch(root_path='res/', idx=iter, info='gt', data_dict=debug_dict,
    # #                    non_plane_idx=-1, save_image=True, save_segmentation=True, save_depth=True,
    # #                    save_cloud=True, save_ply=True)
    #
    # pred
    predict_segmentation[predict_segmentation == (num_queries + 1)] = -1
    predict_segmentation = torch.from_numpy(predict_segmentation)
    inferred_depth = torch.from_numpy(inferred_depth)



    image = image[0].detach().permute(1,2,0).numpy()

    debug_dict = {'image': image,
                  'segmentation': predict_segmentation,
                  'depth_predplane': inferred_depth.detach(),
                  'K_inv_dot_xy_1': k_inv_dot_xy1,}
    visualizationBatch(root_path=r'res/', idx=4, info='planeTR', data_dict=debug_dict,
                       non_plane_idx=-1, save_image=True, save_segmentation=True, save_depth=True,
                       save_cloud=True, save_ply=True)

    # # plane_Seg_Metric = plane_Seg_Metric / len(data_loader)
    #
    # save_path='res/'
    #
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # mine_recalls_pixel = {"PlaneTR (Ours)": pixelDepth_recall_curve * 100}
    # mine_recalls_plane = {"PlaneTR (Ours)": planeDepth_recall_curve[:, 0] / planeDepth_recall_curve[:, 1] * 100}
    # plot_depth_recall_curve(mine_recalls_pixel, type='pixel', save_path=save_path)
    # plot_depth_recall_curve(mine_recalls_plane, type='plane', save_path=save_path)
    #
    # normal_recalls_pixel = {"planeTR": pixelNorm_recall_curve * 100}
    # normal_recalls_plane = {"planeTR": planeNorm_recall_curve[:, 0] / planeNorm_recall_curve[:, 1] * 100}
    # plot_normal_recall_curve(normal_recalls_pixel, type='pixel', save_path=save_path)
    # plot_normal_recall_curve(normal_recalls_plane, type='plane', save_path=save_path)
    #

if __name__ == '__main__':

    eval()

