# 使用少量图片训练的模型对剩下的图像进行标注

from mmdet.apis import init_detector, inference_detector
import mmcv
import mmrotate
import os
import time
import glob
import numpy as np
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from batch_recog_test import NMS, rot_img_360, get_angle
import cv2
import copy
import math

CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', )
cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   }  # in mmdet v2.0 label is 0-based

def obb2poly_oc(cx, cy, w, h, a):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    cosa = math.cos(-a)
    sina = math.sin(-a)
    # print("cos: ", cosa)
    # print("sin: ", sina)
    wx, wy = w * cosa, w * sina
    hx, hy = h * sina, h * cosa
    p1x, p1y = cx - wx - hx, cy + wy - hy
    p2x, p2y = cx + wx - hx, cy - wy - hy
    p3x, p3y = cx + wx + hx, cy - wy + hy
    p4x, p4y = cx - wx + hx, cy + wy + hy
    return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y

def sort_out(dets, rot_dets, labels, img, rot_angle):
    '''图像通过360°矫正后的识别结果排列输出

    Args:
        dets (list): 矫正后的检测框的检测框（仅中心点进行矫正）
        labels (np.array): 检测框的标签
    '''
    # rot_dets只为了得到字符类别的排序和原始坐标的排序
    
    # print('labels: ', labels)
    sort = []
    labels_list = []
    for i in range(10):
    # for i in range(11):
    # for i in range(36):
        for j in range(len(labels[i])):
            labels_list.append(labels[i][j])
    # print('dets: ', len(dets))
    # print('labels_list: ', len(labels_list))
    assert len(dets) == len(labels_list)
    labels_list = np.array(labels_list)
    x = rot_dets[:,0]
    y = rot_dets[:,1]
    # 获取x,y从小到大排序
    indx_x = x.argsort()
    indx_y = y.argsort()
    indx = 0 
    indx_hang1 = []
    indx_hang2 = []
    indx_hang3 = []
    indx = 0
    y_temp = copy.deepcopy(y)
    y_temp -= y_temp[indx_y[0]]
    indx_hang1 = np.where(y_temp < 50)[0]
    # print('indx_y: ', indx_y)
    # 删除第一行的索引
    count = 0
    indx = 0 
    while(count < len(indx_hang1)):
        if indx_y[indx] in indx_hang1:
            count += 1
            indx_y = np.delete(indx_y, indx)
        else:
            indx += 1
    # print('indx_hang1: ', indx_hang1)
    # print('indx_y: ', indx_y)
    if len(indx_y) != 0:
        y_temp = copy.deepcopy(y)
        y_temp -= y_temp[indx_y[0]]
        indx_hang2 = np.setdiff1d(np.where(y_temp < 50)[0], indx_hang1)
        # 删除第二行的索引
        count = 0
        indx = 0 
        while(count < len(indx_hang2)):
            if indx_y[indx] in indx_hang2:
                count += 1
                indx_y = np.delete(indx_y, indx)
            else:
                indx += 1
    indx_hang3 = indx_y
    hang1x_indx = x[indx_hang1].argsort()
    hang2x_indx = x[indx_hang2].argsort()
    hang3x_indx = x[indx_hang3].argsort()
    for i in hang1x_indx:
        sort.append(indx_hang1[i])
    for i in hang2x_indx:
        sort.append(indx_hang2[i])
    for i in hang3x_indx:
        sort.append(indx_hang3[i]) 
    # print('sort: ', sort)
    # print('res_lable: ', labels_list[sort])
    # print('dets:len: ', len(dets[sort]))
    # print('labels:len: ', len(labels_list[sort]))
    
    # 得到排序后的类别和坐标，此时的坐标是原始的，得到排序结果之后rot_dets就没用了
    new_dets = dets[sort]
    new_labels = labels_list[sort]
    # save_txt = '/data1/hzj/mmrotate/data/img_0526/annfiles-label/' + os.path.basename(img).replace('jpg', 'txt')
    save_txt = '/data1/hzj/mmrotate/data/img_0526/annfiles-label/' + os.path.basename(img).replace('png', 'txt')
    print('save_txt: ', save_txt)
    f = open(save_txt, 'a+')
    for i in range(len(new_dets)):
        # print("res: ", str(new_dets[i]) + " " + str(new_labels[i]))
        cx = new_dets[i][0]
        cy = new_dets[i][1]
        w = int(new_dets[i][2] / 2)
        h = int(new_dets[i][3] / 2)
        # print("cxcy: ", [cx, cy, w, h])
        p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = obb2poly_oc(cx, cy, w, h, rot_angle)
        # print("points: ", [int(p1x), int(p1y), int(p2x), int(p2y), int(p3x), int(p3y), int(p4x), int(p4y), new_labels[i]])
        f.write(str(int(p1x)) + " " + str(int(p1y)) + " " + str(int(p2x)) + " " + str(int(p2y)) 
        + " " + str(int(p3x)) + " " + str(int(p3y))+ " " + str(int(p4x)) + " " + str(int(p4y)) + " " + str(new_labels[i]) + " 0\n")
    return 

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

# 钢厂新数据集部分
# 手动标注的242张图像训练对剩下的进行标注
# config_file = '/data2/hzj/mmrotate/work_dirs/img_pre_train/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/img_pre_train_noflip/latest.pth'
# 模型标注且人工矫正后的对所有图像检测
# config_file = '/data2/hzj/mmrotate/work_dirs/img_train_2/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/img_train_2/latest.pth'

# img_0526数据集标注
config_file = '/data2/hzj/mmrotate/work_dirs/img_0526/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
checkpoint_file = '/data2/hzj/mmrotate/work_dirs/img_0526/latest.pth'
dir_path = '/data1/hzj/mmrotate/data/img_0526/images'
# dir_path = '/data1/hzj/mmrotate/data/img_0526/test'

# 钢厂新数据集
# dir_path = '/data1/hzj/mmrotate/data/img_pre_train-dotatxt/images'
# dir_path = '/data1/hzj/mmrotate/data/img_pre_my/images'
# dir_path = '/data1/hzj/mmrotate/data/img_cut/images_all'
# dir_path = '/data1/hzj/mmrotate/data/img_cut/images_2'


# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cpu', cfg_options={'model.test_cfg.score_thr':0.5})
model = init_detector(config_file, checkpoint_file, device='cuda:3', cfg_options={'model.test_cfg.score_thr':0.5})
# model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})

# timers
_t = {'im_detect': Timer(), 'misc': Timer()}

save_path = '/data1/hzj/mmrotate/data/img_0526/images-label'

if not os.path.exists(save_path):
    os.makedirs(save_path)
# img_names = glob.glob(dir_path+'/*bmp')
# img_names = glob.glob(dir_path+'/*jpg')
img_names = glob.glob(dir_path+'/*png')
num_images = len(img_names)
total_time = 0
kong = []
for i, img in enumerate(img_names):
    # test a single image and show the results
    #img = os.path.join(dir_path, img_name)  # or img = mmcv.imread(img), which will only load it once
    _t['im_detect'].tic()
     # 原图
    im = cv2.imread(img)
    # 旋转后的图像
    temp_im = copy.deepcopy(im)
    result = inference_detector(model, img)
    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
    dets = np.vstack(result)
    if len(dets) == 0:
        kong.append(img)
        continue
    rot_dets = copy.deepcopy(dets)
    print('img: ', img)
    print('result: ', dets)
    rot_angle = get_angle(dets)
    rot_dets = rot_img_360(rot_dets, temp_im, rot_angle, img)
        
    #对图像中可能存在的多个类别检测框检测出同一目标进行NMS后处理，只保留confidence最高的那个类别
    nms_dets, tag_warnning, tags_iou = NMS(rot_dets, thresh=0.3)
    # 旋转矫正后的坐标和原始检测得到的坐标
    rot_dets = rot_dets[nms_dets]
    dets = dets[nms_dets]
    # print('rot_dets: ', rot_dets)
    # print('dets: ', dets)   
     
    indx = 0
    all_del = 0 # 记录

    for i in range(10):
    # for i in range(36):
        tag = 0
        if len(result[i]) == 0:
            continue
        else:
            for j in range(len(result[i])):
                if indx not in nms_dets:
                    result[i] = np.delete(result[i], j-tag, 0)
                    labels[i] = np.delete(labels[i], j-tag, 0)
                    tag = tag + 1
                indx += 1
    detect_time = _t['im_detect'].toc(average=False)
    if i > 0:
        total_time += detect_time
    print('len: ', len(dets))
    # 如果检测的字符数量不等于11就不标注，后续手动标
    if len(dets) == 11 or len(dets) == 12:
        # 将检测框进行排序，然后写入标签
        sort_out(dets, rot_dets, labels, img, rot_angle)
        # or save the visualization results to image files
    model.show_result(img, result, out_file=os.path.join(save_path, os.path.basename(img)), bbox_color=(0, 255, 255))

print(f'kong: {kong}')
print("average time: " + str(total_time/(num_images-1)*1000) + ' ms')