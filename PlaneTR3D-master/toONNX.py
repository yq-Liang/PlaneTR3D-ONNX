import torch
import torch.onnx
from models.planeTR_HRNet import PlaneTR_HRNet as PlaneTR
import os
import yaml
import argparse
from easydict import EasyDict as edict
import torchviz

parser = argparse.ArgumentParser()

parser.add_argument('--cfg_path', default='configs/config_planeTR_eval.yaml', type=str,
                    help='full path of the config file')
# parser.add_argument('--cfg_path', default='configs/config_planeTR_train.yaml', type=str,
#                     help='full path of the config file')

args = parser.parse_args()


def pth_to_onnx(input_0,input_1,input_2, checkpoint, onnx_path, cfg,input_names, output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = PlaneTR(cfg) # 导入模型
    # model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cpu')))# 初始化权重
    # model.eval()
    # model.to(device)
    # input=(input_0,input_1,input_2)
    # torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
    #                   output_names=output_names,dynamic_axes={'input_0' : {0 :"batch",2:"height",3:"weight"},    # variable lenght axes
    #                               'output' : {0:"batch" ,2:"height",3:"weight"}},opset_version=11)  # 指定模型的输入，以及onnx的输出路径
    # print("Exporting .pth model to onnx model has been successful!")
    # out=model(input_0,input_1,input_2)
    # print(out)
    # out_list=list(out.values())
    # g=torchviz.make_dot(out['pixel_depth'])
    # g.render('test')
def Set_Config(args):
    # get config file path
    cfg_path = args.cfg_path

    # load config file
    f = open(cfg_path, 'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    cfg = edict(x)

    return cfg

if __name__ == '__main__':
    # 设置环境
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    #路径
    checkpoint = './ckpts/PlaneTR_Pretrained.pt'
    onnx_path = './tinynet.onnx'

    #输入 image b 2
    input_0 = torch.randn(1, 3, 256,192)#b 3 h w

    input_1=torch.randn(1,200,4)#lines

    input_2=torch.randn(1)#num_lines



    # input=(input_0)
    input_names=['input_0','input_1','input_2']
    # input_names=['input_0']
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    cfg=Set_Config(args)

    pth_to_onnx(input_0,input_1,input_2, checkpoint, onnx_path,cfg,input_names)