from __future__ import annotations
from hashlib import new
from operator import gt
import re
from mmdet.apis import init_detector, inference_detector
import mmcv
import mmrotate
import os
import time
import glob
import numpy as np
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
import copy
import math
import cv2
import operator

# CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', )
# CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
           'R','S','T','U','V','W','X','Y','Z', )

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


def load_label(ann_file, version):
    '''
    输入的参数：
    对应检测图像的标签文本
    角度类别：默认为('oc')
    '''  
    cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   } 

    gt_bboxes = []
    gt_labels = []
    gt_polygons = []

    with open(ann_file) as f:
        s = f.readlines()
        for si in s:
            bbox_info = si.split()
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                x, y, w, h, a = poly2obb_np(poly, version)
            except:  # noqa: E722
                continue
            cls_name = bbox_info[8]
            difficulty = int(bbox_info[9])
            label = cls_map[cls_name]
            if difficulty > 100:
                pass
            else:
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)

    return gt_labels

def landm_dets(dets):
    '''
    通过关键点预测结果生产bbox以及角度
    Args:
        dets : 网络预测的结果 bbox中心坐标, 宽, 高, 角度, 左上坐标, 右上坐标, 中心点坐标, 置信度
    '''
    # 预测框的中心点用关键点预测的中心点替换
    dets[:, 0] = dets[:, 9]
    dets[:, 1] = dets[:, 10]
    # 角度
    tl_x = dets[:, 5]
    tl_y = dets[:, 6]
    tr_x = dets[:, 7]
    tr_y = dets[:, 8]
    dx = tr_x - tl_x
    # 图像的y坐标和原点是反的
    dy = tl_y - tr_y
    for j in range(len(dx)):
        if math.atan2(dy[j], dx[j]) > 0:
            dets[j][4] = -math.atan2(dy[j], dx[j]) 
        else:
            dets[j][4] = -(2*np.pi + math.atan2(dy[j], dx[j])) 
    # 此时角度是0-360的变成针对bbox的角度范围
    # for i in range(len(dets)):
    #     if dets[i][4] < np.pi/2:
    #         dets[i][4] = -dets[i][4]
    #     elif dets[i][4] < np.pi:
    #         dets[i][4] = np.pi-dets[i][4]
    #     elif dets[i][4] < 3*np.pi/2:
    #         dets[i][4] = np.pi-dets[i][4]
    #     elif dets[i][4] < 2*np.pi:
    #         dets[i][4] = 2*np.pi-dets[i][4]
    # 宽高
    ctltr_x = (tl_x + tr_x)/2
    ctltr_y = (tl_y + tr_y)/2
    cx = dets[:, 0]
    cy = dets[:, 1]
    dets[:, 2] = pow(pow(tl_x-tr_x,2)+pow(tl_y-tr_y,2),0.5)
    dets[:, 3] = 2* pow(pow(ctltr_x-cx,2)+pow(ctltr_y-cy,2),0.5)
    print('dets_landm: ', dets)
    return dets

def wh_dets(dets):
    '''
    网络得到的预测结果bbox的w可能大于也可能小于h，将其统一调整为w < h
    Args:
        dets (list): 预测的检测结果
    '''
    indx = np.where(dets[:,2] > dets[:,3])
    # print('indx: ', indx)
    if len(indx[0]) == 0:
        return dets
    # 将w, h互换
    else :
        # print(dets[indx, 2])
        temp = dets[indx, 2]
        dets[indx, 2] = dets[indx, 3]
        dets[indx, 3] = temp
        for i in indx[0]:
            if dets[i][4] > 0:
                dets[i][4] = dets[i][4] - np.pi/2
            else:
                dets[i][4] = np.pi/2 + dets[i][4]
    return dets

def get_angle(dets, flag=False):
    '''根据输入的bbox和landm结果，对角度进行360°处理，得到图像旋转的角度

    Args:
        dets (List): 检测结果(bbox(x,y,w,h,theta) + landm(tl_x,tl_y,tr_x,tr_y,cs,cy) + score)
        flag (Bool): 是否根据左上右上直接计算得到的角度对bbox变化得到的角度进行修正
    '''
    # print('dets: ', dets)
    temp_angle = copy.deepcopy(dets[:, 4])
    for i in range(len(dets)):
        # print('111111')
        # 获取左上右上点的中心点
        ctltr_x = (dets[i][5] + dets[i][7])/2
        ctltr_y = (dets[i][6] + dets[i][8])/2
        # print('c_x: ', ctltr_x)
        # print('c_y: ', ctltr_y)
        # 对于字符的整体朝向预测，整体字符框的w取长边
        # if dets[i][4] > 0:
        #     dets[i][4] = dets[i][4] - np.pi/2
        # else:
        #     dets[i][4] = np.pi/2 + dets[i][4]
        
        if ctltr_x < dets[i][0] and ctltr_y < dets[i][1]: 
            # 范围应该0-90°，theta值应该为负，出现正值说明此时可能在90°或者360°附近
            # 如果theta为正，说明检测框稍微超过90°，或者不足0°（360°）
            if dets[i][4] > 0:
                # 如果与中心点x坐标很接近，说明此时检测框不足0°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - 2*np.pi
                # 如果中心点y坐标很接近，说明此时检测框超过90°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi
        elif ctltr_x < dets[i][0] and ctltr_y >= dets[i][1]: 
            # 范围应该在90°-180°之间，theta值应该为正，出现负值说明此时可能在90°或者180°附近
            if dets[i][4] < 0:
                # 如果与中心点x坐标很接近，说明此时检测框超过180°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - np.pi
                # 如果中心点y坐标很接近，说明此时检测框不足90°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4]
            else:
                dets[i][4] = dets[i][4] - np.pi 
        elif ctltr_x >= dets[i][0] and ctltr_y >= dets[i][1]:
            # 范围应该在180°-270°之间，theta值应该为负，出现正值说明此时可能不足180°或超过270°
            if dets[i][4] > 0:
                # 如果与中心点x坐标很接近，说明此时检测框不足180°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4] - np.pi
                # 如果中心点y坐标很接近，说明此时检测框超过270°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi*2
            else:
                dets[i][4] = dets[i][4] - np.pi 
        elif ctltr_x >= dets[i][0] and ctltr_y < dets[i][1]:
            # 范围应该在270°-360°之间，theta值应该为正，出现负值说明此时可能不足270°或超过360°
            if dets[i][4] < 0:
                # 如果与中心点x坐标很接近，说明此时检测框超过360°
                if abs(ctltr_x - dets[i][0]) < 5: 
                    dets[i][4] = dets[i][4]
                # 如果中心点y坐标很接近，说明此时检测框不足270°
                elif abs(ctltr_y - dets[i][1]) < 5:
                    dets[i][4] = dets[i][4] - np.pi
            else:
                dets[i][4] = dets[i][4] - np.pi*2 
        # 用左上右上计算得到的角度进行修正
    if flag:
        tl_x = dets[:, 5]
        tl_y = dets[:, 6]
        tr_x = dets[:, 7]
        tr_y = dets[:, 8]
        dx = tr_x - tl_x
        # 图像的y坐标和原点是反的
        dy = tl_y - tr_y
        for j in range(len(dx)):
            if math.atan2(dy[j], dx[j]) > 0:
                temp_angle[j] = math.atan2(dy[j], dx[j]) 
            else:
                temp_angle[j] = 2*np.pi + math.atan2(dy[j], dx[j]) 
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于5°
            if abs(temp_angle[j] + dets[j][4]) > 0.09:
            # # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于10°
            # if abs(temp_angle[j] + dets[j][4]) > 0.18:
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于20°
            # if abs(temp_angle[j] + dets[j][4]) > 0.35:
            # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于30°
            # if abs(temp_angle[j] + dets[j][4]) > 0.5:
                dets[j][4] = -temp_angle[j]
    # print('angles: ', temp_angle)
    # print('bbox_angles: ', dets[:,4])
       
    # print('rot_angles: ', dets[:,4]/np.pi*180)
    # print('rot_angles: ', dets[:,4])
    
    # 检测个数为偶数个时，求median会出错，它会取两个中间数的平均值
    rot_angles = dets[:, 4].sort()
    # print('rot_angles: ', dets[:,4])
    rot_angle = dets[:,4][len(dets)//2]
    # rot_angle = np.median(dets[:, 4])
    
    # 整体朝向的角度
    # rot_angle = dets[4]
    # rot_angle = np.average(dets[:, 4])
    
    # print('rot_angle: ', rot_angle)
    return rot_angle 

def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 0] + dets[:, 2]
    y2 = dets[:, 1] + dets[:, 3]
    scores = dets[:, -1]

    tags_iou = {} #用于保留所有框与其置信度大于0.5框的对应关系
    tag_warnning = 0 # 出现重叠框的警告
    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]
    # print('areas: ', areas)
    # print('order: ', order)
    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)
        # 得到IoU大于阈值的bbox，
        # inds_1 = np.where(ovr > thresh)[0]
        # print('inds_1: ', inds_1)
        # 如果其置信度大于0.5，仍然保留
        # tag_iou = [] #用于保留当前框置信度大于0.5的重叠框
        # for i in inds_1:
        #     if scores[order[i]] > 0.5:
        #         keep.append(order[i+1])
        #         tag_iou.append(order[i+1])
        #         tag_warnning = 1
        # if len(tag_iou) != 0:
        #     tags_iou[order[0]] = tag_iou
        # print('order: ', order)
        # print('tags_iou: ', tags_iou)
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        #print('order: ', order)
    #返回保留的bbox，以及出现重叠且置信度较大的警告
    return sorted(keep), tag_warnning, tags_iou

def rot_img_360(dets, im, angle, img):
    '''对图像进行0-360°旋转矫正

    Args:
        dets (_type_): _description_
        im (_type_): _description_
        angle (_type_): _description_
        img (_type_): _description_

    Returns:
        _type_: _description_
    '''
    im_w, im_h = im.shape[1], im.shape[0]
    cx, cy = 0.5 * im_w, 0.5 * im_h
    # print('cx: ', cx)
    # print('cy: ', cy)
    # angle = dets[0][4]/math.pi*180 #角度制
    # angle = np.median(dets[:,4])/math.pi*180
    # rot_angle = -dets[0][4] #弧度制
    rot_angle = angle / math.pi*180 #角度制
    # print('angle: ', angle)
    # print('rot_angle: ', rot_angle)
    # 逆时针旋转angle
    M = cv2.getRotationMatrix2D((cx,cy), rot_angle , 1)
    new_w = int(im_h * np.abs(M[0, 1]) + im_w * np.abs(M[0, 0]))
    new_h = int(im_h * np.abs(M[0, 0]) + im_w * np.abs(M[0, 1]))
    # print('new_w: ', new_w)
    # print('new_h: ', new_h)
    # 调整旋转矩阵以考虑平移, 使得整个图像信息不丢失
    M[0, 2] += (new_w - im_w) / 2
    M[1, 2] += (new_h - im_h) / 2
    image_rotation = cv2.warpAffine(src=im, M=M, dsize=(new_w, new_h), borderValue=(255, 255, 255))
    ncx, ncy = 0.5 * image_rotation.shape[1], 0.5 * image_rotation.shape[0]
    # print('ncx: ', ncx)
    # print('ncy: ', ncy)
    # 对旋转后的图像的bbox的中心点也进行相应旋转
    for i in range(len(dets)):
        x = dets[i][0]
        y = dets[i][1]
        dets[i][0] = math.cos(-angle) * (x - cx) - math.sin(-angle) * (y - cy) + ncx
        dets[i][1] = math.sin(-angle) * (x - cx) + math.cos(-angle) * (y - cy) + ncy      
    #     cv2.circle(image_rotation, (int(dets[i][0]), int(dets[i][1])), 5, (0,0,255))
    #     cv2.line(image_rotation, (int(dets[i][0] - dets[i][2]/2), int(dets[i][1] - dets[i][3]/2)),
    #                              (int(dets[i][0] + dets[i][2]/2), int(dets[i][1] - dets[i][3]/2)), (0,0,255))
    # print("dets-rot: ", dets)
    # cv2.imwrite('/data1/hzj/mmrotate/show-dir/recog_wrong/' + osp.basename(img).replace('.bmp','-new.bmp'),image_rotation)

    return dets

def res_out_360(dets, labels):
    '''图像通过360°矫正后的识别结果排列输出

    Args:
        dets (list): 矫正后的检测框的检测框（仅中心点进行矫正）
        labels (np.array): 检测框的标签
    '''
    # print('labels: ', labels)
    sort = []
    x = dets[:,0]
    y = dets[:,1]
    # 获取x,y从小到大排序
    indx_x = x.argsort()
    indx_y = y.argsort()
    # print(f'y: {y}')
    # indx = 0 
    while len(indx_y) != 0:
        # print(f'indx_y: {indx_y}')
        indx_hang = []
        y_temp = copy.deepcopy(y)
        # print(f'y_temp: {y_temp}')
        y_temp -= y_temp[indx_y[0]]
        # print(f'y_temp: {y_temp}')
        indx_hang = np.where(abs(y_temp) < 35)[0]
        # indx_hang = np.where(abs(y_temp) < 50)[0]
        # print(f'indx_hang: {indx_hang}')
        count = 0
        indx = 0 
        while(count < len(indx_hang)):
            # print(f' indx: {indx}')
            if indx_y[indx] in indx_hang:
                count += 1
                y[indx_y[indx]] = 0
                indx_y = np.delete(indx_y, indx)
            else:
                indx += 1
            # print(f'count: {count}')
        hangx_indx = x[indx_hang].argsort()
        for i in hangx_indx:
            sort.append(indx_hang[i])
    
    return labels[sort].tolist(), 0


if __name__ == '__main__':
    # yolox识别性能
    # config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/yolox_s_8x8_300e_GangPei.py'
    # checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/latest.pth'

    # KFIoU-R3det-res18: GPU(ms) CPU(ms) 训练数据为meanspadding + src + 90180270（标签未修改，部分点标注顺序不一致）
    # data/GangPi-random-rotate-meanspadding-src-90180270/test集：acc_rate(95.82%) acc_map0.9_rate(98.51%) 
    # data/GangPi-random-rotate-meanspadding-src-90180270/test集 + NMS：acc_rate(94.93%) acc_map0.9_rate(98.81%)
    # acc_rate: 165/352 46.74%
    # config_file = '/home/zhenjia/zjhu/mmrotate/configs/kfiou/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270-new/latest.pth'

    # KFIoU-R3det-res18-360: GPU(ms) CPU(ms) 训练数据为meanspadding + src + 90180270 （标签经过修改）
    # 预测角度范围为0-360°
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # clsloss : bboxloss : landmarkloss = 1:3:1 accrate: 289/352 82.10%
    # 通过landm直接计算的角度进行纠正得到 / %(5°) / %(10°) /(15°) / %(20°)
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-131/best_acc.pth'
    # clsloss : bboxloss : landmarkloss = 1:5:1  accrate: 315/352 89.49%
    # 通过landm直接计算的角度进行纠正得到 336/352 95.45%(5°) 336/352 95.45%(10°) /352(15°) /352 %(20°)
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/best_acc.pth'
    # clsloss : bboxloss : landmarkloss = 1:7:1  accrate: 311/352 88.35%
    # 通过landm直接计算的角度进行纠正得到 333/352 94.60%(5°) 333/352 94.60%(10°) /352(15°) /352 %(20°)
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-171/best_acc.pth'
    # clsloss : bboxloss : landmarkloss = 1:10:1 accrate: 308/352 87.5%
    # 通过landm直接计算的角度进行纠正得到 319/352  90.63%(5°) 319/352 90.63%(10°) /352(15°) /352 %(20°)
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-1101/best_acc.pth'
    # # 只通过关键点信息计算 279/352 79.04%
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-flip-101/latest.pth'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101/latest.pth'

    # res50  304/352 86.36%
    # 通过landm直接计算的角度进行纠正得到 /352  %(5°) 335/352 95.17%(10°) /352(15°) /352 %(20°)
    # config_file = '/home/zhenjia/zjhu/mmrotate/configs/kfiou/res50_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res50-360test-refine-final-flip/best_acc.pth'

    # r3det + landmark
    # 290/352 82.39%
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_360/r3det_r18_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_360/best_acc.pth'

    # r3det
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/r3det_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/best_acc.pth'
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_Generate_oc/r3det_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_Generate_oc/best_acc.pth'

    # r3det + kfiou + landmarks
    config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-360/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-360/best_acc.pth'

    # Rotate RetinaNet
    # 128 / 352 36.36%
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_retinanet_GangPi_oc/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_retinanet_GangPi_oc/best_acc.pth'
    # save_path = '/data1/hzj/mmrotate/show-dir/rotate-retinanet'
    # 生成数据集
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_RetinaNet_Generate_oc/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_RetinaNet_Generate_oc/best_acc.pth'

    # Rotate Faster_RCNN
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_FasterRCNN_GangPi_le90/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_FasterRCNN_GangPi_le90/best_acc.pth'
    # save_path = '/data1/hzj/mmrotate/show-dir/rotate-faster_rcnn'
    # 生成数据集
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_FasterRCNN_Generate_oc/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_FasterRCNN_Generate_oc/best_acc.pth'

    # RoI Trans
    # 166 / 352 47.16%
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_roi_trans_GangPi_oc/roi_trans_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_roi_trans_GangPi_oc/best_acc.pth'
    # save_path = '/data1/hzj/mmrotate/show-dir/roi_trans'
    # 生成数据集
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_roi_trans_Generate_oc/roi_trans_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_roi_trans_Generate_oc/best_acc.pth'

    # Rotate FCOS
    # 165 / 352 46.875%
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_fcos_r18_fpn_1x_GangPi_le90/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_fcos_r18_fpn_1x_GangPi_le90/best_acc.pth'
    # 生成数据集
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_fcos_Generate_oc/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_fcos_Generate_oc/best_acc.pth'

    # SASM
    # config_file = '/data2/hzj/mmrotate/work_dirs/mmm_GBD_sasm/sasm.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/mmm_GBD_sasm/best_acc.pth'
    
    # Oriented Reppoints
    # config_file = '/data2/hzj/mmrotate/work_dirs/mmm_GBD_oriented_reppoints/oriented_reppoints.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/mmm_GBD_oriented_reppoints/best_acc.pth'
    
    # YOLOX的检测
    # config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/yolox_s_8x8_300e_GangPei.py'
    # checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/latest.pth'

    #YOLOX90180270的检测
    # config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/yolox_8x8_300e_GangPei.py'
    # checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/latest.pth'

    # 不同的损失权重
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-111/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-111/best_acc.pth'
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-112/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-112/best_acc.pth'
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-113/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-113/best_acc.pth'
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-114/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-114/best_acc.pth'

    # 包含整体字符朝向
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/overall-orientation/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/overall-orientation/best_acc.pth'

    # 第一阶段不预测landm，精炼阶段才预测
    # accrate: 315/352 86.65%  311/352 88.35%(5°)
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_360-kfiouRRetinaHead/best_acc.pth'
    
    
    # 0418这个数据集
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/r3det_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-roitrans/roi_trans_r18_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-roitrans/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFasterRCNN/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFasterRCNN/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFCOS/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFCOS/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotRetinanet/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotRetinanet/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-kfiou/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-kfiou/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/r3det_r18_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/mmm_0418_sasm/sasm.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/mmm_0418_sasm/best_acc.pth'
    # config_file = '/data2/hzj/mmrotate/work_dirs/mmm_0418_oriented_reppoints/oriented_reppoints.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/mmm_0418_oriented_reppoints/best_acc.pth'

    # 合成数据集
    # config_file = '/data2/hzj/mmrotate/work_dirs/r3det-res18-360-generate_data/r3det_r18_fpn_1x_GangPi_360.py'
    # checkpoint_file = '/data2/hzj/mmrotate/work_dirs/r3det-res18-360-generate_data/best_acc.pth'
    # config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-oc/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
    # checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-oc/best_acc.pth'

    # build the model from a config file and a checkpoint file
    #model = init_detector(config_file, checkpoint_file, device='cpu', cfg_options={'model.test_cfg.score_thr':0.5})
    model = init_detector(config_file, checkpoint_file, device='cuda:0',  cfg_options={'model.test_cfg.score_thr':0.5})
    #model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    # image-split中的val数据集中的前5张图片
    # dir_path = '/home/zjhu/mm/mmrotate/data/rotate_map_test-5'
    # save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-5-res18'

    #测试不区分类别的NMS以及离群筛除
    # dir_path = '/home/zjhu/mm/mmrotate/data/NMS-test'
    # save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS_test-r3det-res18-kfiou'


    # dir_path = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/test/images'
    # dir_path = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/val/images'
    # dir_path = '/data1/hzj/mmrotate/data/GangPi_dataloader_test/images'
    # dir_path = '/data1/hzj/mmrotate/data/recog-test'
    # dir_path = '/data1/hzj/mmrotate/data/recog-test-1'
    # dir_path = '/data1/hzj/mmrotate/data/recog-test-2'
    # dir_path = '/data1/hzj/mmrotate/data/recog_wrong'

    # 合成数据集
    dir_path = '/data1/hzj/mmrotate/data/generate_data/test/images'

    # 0418数据集
    # dir_path = '/data1/hzj/mmrotate/data/img_0418/test/images'

    # save_path = '/home/zjhu/mm/mmrotate/show-dir/meanspadding-src-90180270-mAP-r3det-CSPDarknet-kfiou'
    # save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS-meanspadding-src-90180270-mAP-r3det-CSPDarknet-kfiou'
    # save_path = '/home/zjhu/mm/mmrotate/show-dir/recog-out-r3det-res18-kfiou' #识别结果输出
    # save_path = '/data1/hzj/mmrotate/show-dir/recog-out-r3det-res18-kfiou-sort'
    # save_path = '/data1/hzj/mmrotate/show-dir/recog-out-test-1-360-w<h'
    save_path = '/data1/hzj/mmrotate/data/img_0418'


# if __name__ == '__main__':

    # 对各种模型进行识别，然后输出图片识别的数字编号
    img_names = glob.glob(dir_path+'/*jpg')
    # img_names = glob.glob(dir_path+'/*bmp')
    num_images = len(img_names)
    total_time = 0
    acc_count = 0
    for i, img in enumerate(img_names):
        # test a single image and show the results
        print('img: ', img)
        # 原图
        im = cv2.imread(img)
        # 旋转后的图像
        temp_im = copy.deepcopy(im)
        _t['im_detect'].tic()
        result = inference_detector(model, img)
        # print('result: ', result)
        result_temp = copy.deepcopy(result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels_list = []
        for label in labels:
            labels_list.extend(list(label))
        labels_list = np.array(labels_list)
        # print('labels: ', labels)
        # print(gt_labels)
        dets = np.vstack(result_temp)
        # print('dets: ', dets)
        if len(dets) == 0:
            continue
        # 将检测结果完全根据关键点信息得到的，通过关键点信息得到检测框
        # dets = landm_dets(dets)
        # print('dets: ', dets)
        
        # 将dets都调整为 w < h
        dets = wh_dets(dets)
        # print('w<h_dets: ', dets)
        
        # 获取360°范围的旋转角度
        # rot_angle = get_angle(np.array([dets[-1]]))
        rot_angle = get_angle(dets)
        # rot_angle = get_angle(dets, True)
        # print('dets: ', dets)
        # rot_angle = dets[-1][4]
        
        # 将图片进行旋转
        dets = rot_img_360(dets, temp_im, rot_angle, img)
        # print(f'dets: {dets}')
        # dets = rot_img_360(dets, temp_im, np.median(dets[:,4]), img)
        
        #对图像中可能存在的多个类别检测框检测出同一目标进行NMS后处理，只保留confidence最高的那个类别
        nms_dets, tag_warnning, tags_iou = NMS(dets, thresh=0.3)
        # nms_dets, tag_warnning, tags_iou = NMS(dets[:-1], thresh=0.3)
        # print('nms_dets: ', nms_dets)
        dets = dets[nms_dets]
        labels = labels_list[nms_dets]
    
        acc_tag = acc_count
        recog_result, tag_warnning_1 = res_out_360(dets, labels)
        detect_time = _t['im_detect'].toc(average=False)
        # print(f'detect_time: {detect_time}')
        # print('recog_result: ', recog_result)
        # 识别准确率计算
        # 识别的GT
        # ann_file = img.replace('bmp', 'txt').replace('images', 'annfiles')
        ann_file = img.replace('jpg', 'txt').replace('images', 'annfiles')
        gt_labels = load_label(ann_file, 'oc')
        if len(recog_result) == 0:
            print('kong: ', img)
        # print('tag_warnning_1: ', tag_warnning_1)
        if tag_warnning_1 != 1 and tag_warnning_1 != 2:
            if operator.eq(gt_labels, recog_result):
                acc_count = acc_count + 1
        else:
            num_images = num_images - 1

        if acc_tag == acc_count:
            # print('img: ', img)
            print('recog_result: ', recog_result)
            print('gt_labels: ', gt_labels)
            # print('rot_angle: ', rot_angle/ math.pi*180)
        # print(type(recog_result))


        if i > 0:
            total_time += detect_time
        # or save the visualization results to image files
        # model.show_result(img, result, out_file=save_img_path)
        # model.show_result(img, result_ori, out_file=save_img_path.replace('.bmp', '-ori.bmp'))
        # model.show_result(img, result, out_file=os.path.join(save_path, 'nms-'+os.path.basename(img)))
    
    print(acc_count)
    print(num_images)
    print(acc_count/(num_images) * 100, '%')
    print(f'{(total_time/(num_images)*1000)}ms')
