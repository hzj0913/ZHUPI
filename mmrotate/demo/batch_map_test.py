from mmdet.apis import init_detector, inference_detector
import math
import os
import glob
import numpy as np
import copy
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np



# CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
           'R','S','T','U','V','W','X','Y','Z', )
cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   }  # in mmdet v2.0 label is 0-based


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

def dets_landm(result):
    '''用关键点信息对bbox进行矫正

    Args:
        dets (_type_): _description_
    '''
    dets = np.vstack(result)
    dets = wh_dets(dets)
    temp_angle = copy.deepcopy(dets[:, 4])
    print('dets: ', dets)
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
            temp_angle[j] = math.atan2(dy[j], dx[j]) 
        else:
            temp_angle[j] = 2*np.pi + math.atan2(dy[j], dx[j]) 
        # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于5°
        if abs(temp_angle[j] + dets[j][4]) > 0.15:
        # bbox的360°角度和直接根据左上右上点计算得到的角度差距大于30°
        # if abs(temp_angle[j] + dets[j][4]) > 0.5:
                dets[j][4] = -temp_angle[j]
    indx = 0
    for i in range(10):
        for j in range(len(result[i])):
            result[i][j][2] = dets[indx][2]
            result[i][j][3] = dets[indx][3]
            result[i][j][4] = dets[indx][4]
            indx += 1
    print('dets-new: ', dets)
    return result

def dets_landm_only(result):
    '''只用关键点信息得到bbox信息

    Args:
        result : 检测结果
    '''
    dets = np.vstack(result)
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
    for i in range(len(dets)):
        if dets[i][4] < np.pi/2:
            dets[i][4] = -dets[i][4]
        elif dets[i][4] < np.pi:
            dets[i][4] = np.pi-dets[i][4]
        elif dets[i][4] < 3*np.pi/2:
            dets[i][4] = np.pi-dets[i][4]
        elif dets[i][4] < 2*np.pi:
            dets[i][4] = 2*np.pi-dets[i][4]
    # 宽高
    ctltr_x = (tl_x + tr_x)/2
    ctltr_y = (tl_y + tr_y)/2
    cx = dets[:, 0]
    cy = dets[:, 1]
    dets[:, 2] = pow(pow(tl_x-tr_x,2)+pow(tl_y-tr_y,2),0.5)
    dets[:, 3] = 2* pow(pow(ctltr_x-cx,2)+pow(ctltr_y-cy,2),0.5)
    print('dets_landm: ', dets)
    indx = 0
    for i in range(10):
        for j in range(len(result[i])):
            result[i][j][0] = dets[indx][0]
            result[i][j][1] = dets[indx][1]
            result[i][j][2] = dets[indx][2]
            result[i][j][3] = dets[indx][3]
            result[i][j][4] = dets[indx][4]
            indx += 1
    print('result-new: ', result)
    return result

def get_ann(ann_file):
    gt_bboxes = []
    gt_labels = []
    annotations = {}
    with open(ann_file) as f:
        s = f.readlines()
        for si in s:
            bbox_info = si.split()
            poly = np.array(bbox_info[:8], dtype=np.float32)
            try:
                # print('version: ', self.version)
                x, y, w, h, a = poly2obb_np(poly, 'oc')
                # print('a: ', a)
            except:  # noqa: E722
                continue
            cls_name = bbox_info[8]
            label = cls_map[cls_name]
            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(label)

    annotations['bboxes'] = np.array(
        gt_bboxes, dtype=np.float32)
    annotations['labels'] = np.array(
        gt_labels, dtype=np.int64)

    annotations['bboxes_ignore'] = np.zeros(
        (0, 5), dtype=np.float32)
    annotations['labels_ignore'] = np.array(
        [], dtype=np.int64)

    return annotations

def evaluate(results,
             annotations,
             metric='mAP',
             logger=None,
             proposal_nums=(100, 300, 1000),
             iou_thr=0.5,
             scale_ranges=None,
             nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        print('11111111111')
        nproc = 4
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            # print('2222222222')
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results


# YOLOX的检测
# config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/yolox_s_8x8_300e_GangPei.py'
# checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/yolox'

#YOLOX90180270的检测
# config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/yolox_8x8_300e_GangPei.py'
# checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/yolox-MultiAngle'

# KFIoU-R3det-Res18:
# config_file = '/home/zhenjia/zjhu/mmrotate/configs/kfiou/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270-new/latest.pth'

# R3det-Res18
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/r3det_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/r3det-GangPi'
# 生成数据集
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_Generate_oc/r3det_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_Generate_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/r3det-generate'

# R3det-Res18 + landmark
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_360/r3det_r18_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_360/latest.pth'

# Rotate RetinaNet
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_retinanet_GangPi_oc/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_retinanet_GangPi_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-retinanet'
# 生成数据集
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_RetinaNet_Generate_oc/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_RetinaNet_Generate_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-RetinaNet-generate'

# Rotate Faster_RCNN
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_FasterRCNN_GangPi_le90/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_FasterRCNN_GangPi_le90/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-faster_rcnn'
# 生成数据集
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_FasterRCNN_Generate_oc/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_FasterRCNN_Generate_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-FasterRCNN-generate'

# RoI Trans
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_roi_trans_GangPi_oc/roi_trans_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_roi_trans_GangPi_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/roi_trans'
# 生成数据集
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_roi_trans_Generate_oc/roi_trans_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_roi_trans_Generate_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-RoI-trans-generate'

# Rotate FCOS
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_fcos_r18_fpn_1x_GangPi_le90/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotated_fcos_r18_fpn_1x_GangPi_le90/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate_fcos'
# 生成数据集
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_fcos_Generate_oc/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/rotate_fcos_Generate_oc/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/rotate-fcos-generate'

# 测试360检测器
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-131/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-131/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-171/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-1101/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101/latest.pth'

# 生成数据集的kfiou + landmark
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-360/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-360/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/r3det-kfiou-landmark-generate'

# 不同的损失权重
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-111/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-111/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-112/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-112/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-113/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-113/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-114/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-114/latest.pth'

# 0418数据集
config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/best_acc.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/r3det_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/best_acc.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-roitrans/roi_trans_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-roitrans/latest.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFasterRCNN/rotated_faster_rcnn_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFasterRCNN/latest.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFCOS/rotated_fcos_r18_fpn_1x_GangPi_le90.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotFCOS/latest.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotRetinanet/rotated_retinanet_obb_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-rotRetinanet/latest.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/r3det_r18_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/latest.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-kfiou/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-kfiou/latest.pth'


# 合成数据集
# config_file = '/data2/hzj/mmrotate/work_dirs/r3det-res18-360-generate_data/r3det_r18_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/r3det-res18-360-generate_data/latest.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-oc/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_oc-generatedata-oc/latest.pth'

# dir_path = '/data1/hzj/mmrotate/data/360_test_train/images'
# dir_path = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/test/images'

# 0418数据集
dir_path = '/data1/hzj/mmrotate/data/img_0418/test/images'

# r3det + kfiou test集检测结果
# save_path = '/data1/hzj/mmrotate/show-dir/oc-test'
# save_path = '/data1/hzj/mmrotate/show-dir/360-test-refine-final'
# save_path = '/data1/hzj/mmrotate/show-dir/360-test-refine-final-flip-101'

# 合成数据集 
# dir_path = '/data1/hzj/mmrotate/data/generate_data/test/images'


# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cpu', cfg_options={'model.test_cfg.score_thr':0.5})
model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})
# model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})


# dir_path = '/data1/hzj/mmrotate/data/recog-test-1'
# save_path = '/data1/hzj/mmrotate/show-dir/recog-test-1-360-final'
# save_path = '/data1/hzj/mmrotate/show-dir/recog-test-1-360-final-flip'
# save_path = '/data1/hzj/mmrotate/show-dir/recog-test-1-360-final-flip-171'
# save_path = '/data1/hzj/mmrotate/show-dir/recog-test-1-oc'
# dir_path = '/home/zjhu/mm/mmrotate/data/GangPI_randomAngle/test/images-split/images'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/GangPi_random_rotate_res18-256'
# dir_path = '/home/zjhu/mm/mmrotate/data/GangPei_MultiAngle_test'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/GangPi_MultiAngle_test_r3det'
# dir_path = '/home/zjhu/mm/mmrotate/data/rotate_map_test-676'
# dir_path = '/data1/hzj/mmrotate/data/recog-test-1'
save_path = '/data1/hzj/mmrotate/show-dir/r3det-res18-360'

if __name__ == '__main__':
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # img_names = glob.glob(dir_path+'/*bmp')
    img_names = glob.glob(dir_path+'/*jpg')
    num_images = len(img_names)
    total_time = 0
    results = []
    annotations = []
    for i, img in enumerate(img_names):
        # test a single image and show the results
        #img = os.path.join(dir_path, img_name)  # or img = mmcv.imread(img), which will only load it once
        print('img: ', img)
        result = inference_detector(model, img)
        # print('result: ', result)
        # 用关键点信息将bbox的预测结果进行矫正
        # result = dets_landm(result)
        # 只用关键点信息来生成bbox信息
        # result = dets_landm_only(result)
        results.append(result)
        # annotation = get_ann(img.replace('bmp', 'txt').replace('images', 'annfiles'))
        annotation = get_ann(img.replace('jpg', 'txt').replace('images', 'annfiles'))
        annotations.append(annotation)
        # or save the visualization results to image files
        # model.show_result(img, result, out_file=os.path.join(save_path, os.path.basename(img)))

    eval_result = evaluate(results, annotations, iou_thr=0.5)
    print(eval_result)