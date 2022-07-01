import torch
import torch.nn.functional as F
import numpy as np

def Res(plane_prob,plane_embedding,pixel_embedding,x):
    _,_,ho,wo=x.shape
    num_queries=20
    embedding_dist_threshold=1.0

    pred_logits = plane_prob[-1][0]
    pred_plane_embedding = plane_embedding[-1]
    pred_pixel_embedding = pixel_embedding

    pred_prob = F.softmax(pred_logits, dim=-1)  # num_queries, 3
    print(pred_prob.shape)
    score, labels = pred_prob.max(dim=-1)
    labels[labels != 1] = 0
    # label_mask = labels > 0
    label_mask=torch.where(labels>0,1,0)

    if sum(label_mask) == 0:
        _, max_pro_idx = pred_prob[:, 1].max(dim=0)
        label_mask[max_pro_idx] = 1
    print(label_mask.shape)
    print(pred_plane_embedding.shape)
    valid_plane_embedding = pred_plane_embedding[:,label_mask, :]#1 1 20 8

    c_embedding = pred_plane_embedding.shape[-1]#2
    flat_pixel_embedding = pred_pixel_embedding.view(c_embedding, -1).t()  # hw, c_embedding
    dist_map_pixel2planes = torch.cdist(flat_pixel_embedding, valid_plane_embedding, p=2)# 1 1  1024 20
    #修改
    # flat_pixel_embedding = pred_pixel_embedding.view(1,c_embedding,1, -1).repeat(1,1,20,1).permute(0,3,2,1)#1 hw,1, c_embedding
    # U 1,20,8,1024 ↑ ↓
    # valid_plane_embedding=valid_plane_embedding.repeat(1,ho*wo,1,1)
    # TODO unity处理tensor数据; 或者解决以下代码转成onnx在unity运行 目前似乎两个方法都不行
    # dist_map_pixel2planes=torch.sub(flat_pixel_embedding,valid_plane_embedding)
    # dist_map_pixel2planes=torch.sqrt(torch.sum((flat_pixel_embedding-valid_plane_embedding)**2,3)).unsqueeze(0)
    # dist_map_pixel2planes=torch.sum(torch.abs(torch.sub(flat_pixel_embedding,valid_plane_embedding)),3).unsqueeze(0)

    dist_pixel2onePlane, planeIdx_pixel2onePlane = dist_map_pixel2planes.min(-1)  # [hw,]
    dist_pixel2onePlane = dist_pixel2onePlane.view(ho, wo)  # h, w
    planeIdx_pixel2onePlane = planeIdx_pixel2onePlane.view(ho, wo)  # h, w
    mask_pixelOnPlane = dist_pixel2onePlane <= embedding_dist_threshold  # h, w
    predict_segmentation = planeIdx_pixel2onePlane.cpu().numpy().copy()  # h, w

    if int(mask_pixelOnPlane.sum()) < (ho * wo):  # set plane idx of non-plane pixels as num_queries + 1
        predict_segmentation[mask_pixelOnPlane.cpu() == 0] = num_queries + 1
    predict_segmentation = predict_segmentation.reshape(ho,
                                                        wo)  # h, w (0~num_queries-1:plane idx, num_queries+1:non-plane)
    predict_segmentation[predict_segmentation == (num_queries + 1)] = -1

    segmentation =torch.from_numpy(predict_segmentation)  # -1 indicates non-plane region
    # assert segmentation.dim() == 3 or segmentation.dim() == 2
    if segmentation.dim() == 3:
        assert segmentation.shape[0] == 1
        segmentation = segmentation[0].cpu().numpy()
    else:
        segmentation = segmentation.cpu().numpy()
    # print(np.unique(segmentation))
    segmentation += 1
    colors = torch.from_numpy(labelcolormap(256))
    # ***************  get color segmentation
    seg = torch.stack([colors[segmentation, 0], colors[segmentation, 1], colors[segmentation, 2]], axis=2)
    # ***************  get blend image
    x=x.squeeze(0).permute(1,2,0).numpy()
    blend_seg = (seg * 0.7 + x* 0.3)
    seg_mask = (segmentation > 0).astype(np.uint8)
    seg_mask =seg_mask[:, :, np.newaxis]
    blend_seg = blend_seg * seg_mask + x.astype(np.uint8) * (1 - seg_mask)

    return blend_seg.numpy()

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

import sys

if __name__=='__main__':
    print(Res(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]))