import onnx
import torch
from result import Res

onnx_model = onnx.load("PlaneTR_test.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("Model incorrect")
else:
    print("Model correct")

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import onnxruntime
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as tf
transforms = tf.Compose([
        tf.ToTensor(),
        # tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# data_path='./res/test.png'
data_path='./res/test32.png'
# data_path='face.png'
image=cv2.imread(data_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)# 192 256 3

# #大的可以
image=Image.fromarray(image)
image=transforms(image)#3 h w
image=image.unsqueeze(0)
input=image.numpy()

#原文方法
# input = image.astype(np.float32)
# input = torch.tensor(input)
# input = input.unsqueeze(1)
# # print(input.shape)
# input = input.permute(1, 3, 0, 2)
# image = input.to(device)
# input=input.numpy()

b,_,h,w=image.shape
input_1=torch.linspace(1,h//16,h//16).unsqueeze(1)
input_2=torch.linspace(1,w//16,w//16).unsqueeze(0)
input1=input_1.numpy()
input2=input_2.numpy()
# image_ori = image.copy()
# image = Image.fromarray(image)

# input=torch.Tensor(image)
# print(type(input))
ort_session=onnxruntime.InferenceSession('PlaneTR_test.onnx')
ort_inputs={'input_0':input}
# ort_inputs={'input_0':input,'input_1':input1,'input_2':input2}
outputs=ort_session.run(None,ort_inputs)

path=r'res/check_file_ori.txt'
# path2=r'output.png'
# image=Image.fromarray(np.array(outputs)[0])
# cv2.imwrite(path2, image)

file=open(path,'w+')
for i in range(len(outputs)):
    print(outputs[i].shape)
    file.write(str(i))
    file.write(":\n")
    file.write(str(outputs[i]))
    file.write("\n")
file.close()

# #TODO 打印结果
# res=Res(output1,output2,output3,image)
# blend_seg_path = "test.jpg"
# cv2.imwrite(blend_seg_path, res)

from change_utils.disp import plot_depth_recall_curve, plot_normal_recall_curve, visualizationBatch
import torch.nn.functional as F
from change_utils.misc import get_coordinate_map
import torch


num_queries=20
embedding_dist_threshold=1.0
k_inv_dot_xy1 = get_coordinate_map(device)#hw是否要改
#
# decompose outputs
pred_logits = torch.from_numpy(outputs[0][0])  # num_queries, 3 #不同
# print(1,pred_logits.shape)
pred_param = torch.from_numpy(outputs[1][0])  # num_queries, 3
# print(2,pred_param.shape)
pred_plane_embedding = torch.from_numpy(outputs[2][0])  # num_queries, 2
# print(3,pred_plane_embedding.shape)
pred_pixel_embedding = torch.from_numpy(outputs[3][0])  # 2, h, w
# print(4,pred_pixel_embedding.shape)
pred_pixel_depth = torch.from_numpy(outputs[6][0])  # h, w
# print(5,pred_pixel_depth.shape)

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
depth_maps_inv = torch.matmul(valid_param.to(device), k_inv_dot_xy1)
depth_maps_inv = torch.clamp(depth_maps_inv, min=0.1, max=1e4)
depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)  6 49152
inferred_depth = depth_maps.t()[range(h * w), planeIdx_pixel2onePlane.view(-1)].view(h, w)
inferred_depth = inferred_depth.to(device) * mask_pixelOnPlane.float().to(device) + pred_pixel_depth.to(device) * (1-mask_pixelOnPlane.float().to(device))

# get depth maps
# gt_depth = gt_depth.cpu().numpy()[0, 0].reshape(h, w)  # h, w
inferred_depth = inferred_depth.cpu().numpy().reshape(h, w)
inferred_depth = np.clip(inferred_depth, a_min=1e-4, a_max=10.)

debug_dir='res/'

# # GT
# gt_seg[gt_seg == 20] = -1
# gt_seg = torch.from_numpy(gt_seg)
# gt_depth = torch.from_numpy(gt_depth)
# debug_dict = {'image': image[0].detach(), 'segmentation': gt_seg.detach(),
#               'depth_GT': gt_depth.detach(),
#               'K_inv_dot_xy_1': k_inv_dot_xy1, }
# visualizationBatch(root_path=debug_dir, idx=iter, info='gt', data_dict=debug_dict,
#                    non_plane_idx=-1, save_image=True, save_segmentation=True, save_depth=True,
#                    save_cloud=True, save_ply=True)

# pred
predict_segmentation[predict_segmentation == (num_queries + 1)] = -1
predict_segmentation = torch.from_numpy(predict_segmentation)
inferred_depth = torch.from_numpy(inferred_depth)

image=image[0].detach().permute(1,2,0).cpu().numpy()

debug_dict = {'image': image,
              'segmentation': predict_segmentation,
              'depth_predplane': inferred_depth,
              'K_inv_dot_xy_1': k_inv_dot_xy1, }
visualizationBatch(root_path=debug_dir, idx=3, info='planeTR', data_dict=debug_dict,
                   non_plane_idx=-1, save_image=True, save_segmentation=True, save_depth=True,
                   save_cloud=True, save_ply=True)
