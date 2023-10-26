from mmdet.apis import init_detector, inference_detector
import mmcv
import mmrotate
import os
import time
import glob
import numpy as np
import os.path as osp
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np


# CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', )
CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
# CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
#            'R','S','T','U','V','W','X','Y','Z', )
cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   }  # in mmdet v2.0 label is 0-based

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


# YOLOX的检测
# config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/yolox_s_8x8_300e_GangPei.py'
# checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/yolox'

#YOLOX90180270的检测
# config_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/yolox_8x8_300e_GangPei.py'
# checkpoint_file = '/data1/hzj/mmdetection/work-dir/yolox_s_8x8_300e_GangPei_MulitAngle/latest.pth'
# save_path = '/data1/hzj/mmrotate/show-dir/yolox-MultiAngle'

# 任意旋转角度的检测
# KFIOU-R3det: GPU(110.66ms) CPU(1694.31ms)
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270-new/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270-new/latest.pth'
# save_path = 

# R3det-CSPDarknet: GPU(85.048ms) CPU(1032.30ms)
# config_file = 'work-dir/r3det-CSPDarknet/r3det-CSPDarknet_In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-CSPDarknet/epoch_40.pth'

# KFIoU-R3det-CSPDarknet: GPU(85.27ms) CPU(1045.43ms) 
# config_file = 'work-dir/r3det-kfiou-CSPDarknet/r3det_CSPDarknet_kfiou_In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-CSPDarknet/epoch_40.pth'

# KFIoU-R3det-CSPDarknet: GPU(78.11ms) CPU(ms) 训练数据为meanspadding + src + 90180270
# config_file = 'work-dir/r3det-kfiou-resnet18-meanspadding-src-90180270/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18-meanspadding-src-90180270/epoch_40.pth'

# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270/epoch_40.pth'

# KFIoU-R3det-Res18: GPU(86.24ms) CPU(1051.25ms)
# config_file = 'work-dir/r3det-kfiou-resnet18/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18/epoch_40.pth'

# KFIoU-R3det-Res18-512: GPU(42.04ms) CPU(283.84ms) 输入图像尺寸为512
# config_file = 'work-dir/r3det-kfiou-resnet18-512/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18-512/epoch_40.pth'

#KFIoU-R3det-Res18-256: GPU(38.10ms) CPU(113.45ms) 输入图像尺寸为256
# config_file = 'work-dir/r3det-kfiou-resnet18-256/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18-256/epoch_40.pth'

# 生成数据集上测试
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_360-generatedata/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/res18_kfiou_ln_r50_fpn_1x_GangPi_360-generatedata/latest.pth'

# 测试360检测器
# base
# config_file = '/home/zhenjia/zjhu/mmrotate/configs/kfiou/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det-kfiou-resnet18-meanspadding-src-90180270-new/latest.pth'
# 有refine
# config_file = '/home/zhenjia/zjhu/mmrotate/configs/kfiou/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-1/epoch_40.pth'
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-filp-171/latest.pth'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101/latest.pth'

# R3det + res18
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/r3det_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/r3det_r18_fpn_1x_GangPi_oc/latest.pth'

# 包含整体字符朝向
# config_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/overall-orientation/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data1/hzj/mmrotate/work_dirs/work_dirs/overall-orientation/latest.pth'

# 钢厂新数据集部分
# config_file = '/data2/hzj/mmrotate/work_dirs/img_pre_train/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/img_pre_train_noflip/latest.pth'

# dir_path = '/data1/hzj/mmrotate/data/360_test_train/images'
# dir_path = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/test/images'
# 钢厂新数据集
# dir_path = '/data1/hzj/mmrotate/data/img_pre_train-dotatxt/images'



# 0418数据集 
dir_path = '/data1/hzj/mmrotate/data/img_0418/test/images'
save_path = '/data2/hzj/mmrotate/show-dir/exp0418-r3det360'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3detkfiou360/best_acc.pth'
# config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/r3det_r18_fpn_1x_GangPi_oc.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det/best_acc.pth'
config_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/r3det_r18_fpn_1x_GangPi_360.py'
checkpoint_file = '/data2/hzj/mmrotate/work_dirs/exp0418-r3det-360/latest.pth'


# r3det + kfiou test集检测结果
# save_path = '/data1/hzj/mmrotate/show-dir/oc-test'
# save_path = '/data1/hzj/mmrotate/show-dir/360-test-refine-final'
# save_path = '/data1/hzj/mmrotate/show-dir/360-test-refine-final-flip-101'

# dir_path = '/data1/hzj/mmrotate/data/generate_data/test/images'
# save_path = '/data1/hzj/mmrotate/show-dir/generate_data_base'

# 0526数据集
# dir_path = '/data1/hzj/mmrotate/data/img_0526/test'
# save_path = '/data2/hzj/mmrotate/show-dir/img_0526_test'
# config_file = '/data2/hzj/mmrotate/work_dirs/img_0526_all/res18_kfiou_ln_r50_fpn_1x_GangPi_360.py'
# checkpoint_file = '/data2/hzj/mmrotate/work_dirs/img_0526_all/epoch_40.pth'

# build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cpu', cfg_options={'model.test_cfg.score_thr':0.5})
model = init_detector(config_file, checkpoint_file, device='cuda:3', cfg_options={'model.test_cfg.score_thr':0.5})
# model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})

# timers
_t = {'im_detect': Timer(), 'misc': Timer()}

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
# save_path = '/data1/hzj/mmrotate/show-dir/overall-orientation'



# save_path = '/data2/hzj/mmrotate/show-dir/img_pre_train'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# img_names = glob.glob(dir_path+'/*bmp')
img_names = glob.glob(dir_path+'/*jpg')
# img_names = glob.glob(dir_path+'/*png')
num_images = len(img_names)
total_time = 0
for i, img in enumerate(img_names):
    # test a single image and show the results
    #img = os.path.join(dir_path, img_name)  # or img = mmcv.imread(img), which will only load it once
    _t['im_detect'].tic()
    result = inference_detector(model, img)
    dets = np.vstack(result)
    print('img: ', img)
    # print('result: ', dets)
    detect_time = _t['im_detect'].toc(average=False)
    if i > 0:
        total_time += detect_time
    # or save the visualization results to image files
    model.show_result(img, result, out_file=os.path.join(save_path, os.path.basename(img)), bbox_color=(0, 255, 255))

print("average time: " + str(total_time/(num_images-1)*1000) + ' ms')