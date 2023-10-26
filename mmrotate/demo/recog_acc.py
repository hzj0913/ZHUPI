from __future__ import annotations
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


CLASSES = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )

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


def load_ann(ann_file, version):
    '''
    输入的参数：
    对应检测图像的标签文本
    角度类别：默认为('oc')
    '''  
    cls_map = {c: i
                   for i, c in enumerate(CLASSES)
                   } 

    data_info = {}
    img_id = osp.split(ann_file)[1][:-4]
    # img_name = img_id + '.png'
    img_name = img_id + '.bmp'
    data_info['filename'] = img_name
    data_info['ann'] = {}
    gt_bboxes = []
    gt_labels = []
    gt_polygons = []
    gt_bboxes_ignore = []
    gt_labels_ignore = []
    gt_polygons_ignore = []

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

    if gt_bboxes:
        data_info['ann']['bboxes'] = np.array(
            gt_bboxes, dtype=np.float32)
        data_info['ann']['labels'] = np.array(
            gt_labels, dtype=np.int64)
        data_info['ann']['polygons'] = np.array(
            gt_polygons, dtype=np.float32)
    else:
        data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                dtype=np.float32)
        data_info['ann']['labels'] = np.array([], dtype=np.int64)
        data_info['ann']['polygons'] = np.zeros((0, 8),
                                                dtype=np.float32)

    if gt_polygons_ignore:
        data_info['ann']['bboxes_ignore'] = np.array(
            gt_bboxes_ignore, dtype=np.float32)
        data_info['ann']['labels_ignore'] = np.array(
            gt_labels_ignore, dtype=np.int64)
        data_info['ann']['polygons_ignore'] = np.array(
            gt_polygons_ignore, dtype=np.float32)
    else:
        data_info['ann']['bboxes_ignore'] = np.zeros(
            (0, 5), dtype=np.float32)
        data_info['ann']['labels_ignore'] = np.array(
            [], dtype=np.int64)
        data_info['ann']['polygons_ignore'] = np.zeros(
            (0, 8), dtype=np.float32)

    #self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
    return data_info

def evaluate(results,
             annotation,
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
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        #annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotation,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                dataset=CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        return eval_results

def NMS(dets, thresh):
    """Pure Python NMS baseline."""
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 0] + dets[:, 2]
    y2 = dets[:, 1] + dets[:, 3]
    scores = dets[:, 5]

    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]
    print('areas: ', areas)
    print('order: ', order)
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
        #print('inds: ', inds)
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
        #print('order: ', order)
    return sorted(keep)

# 任意旋转角度的检测
# KFIOU-R3det: 
# VAL集: acc_rate(99.11%) acc_map0.9_rate(99.85%) 
# 90,180,270 test集: acc_rate(33.54%) acc_map0.9_rate(70.73%) 
# config_file = 'work-dir/r3det_kfiou_In_r50_fpn_1x_GangPi_oc/r3det_kfiou_In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det_kfiou_In_r50_fpn_1x_GangPi_oc/epoch_40.pth'

# R3det-CSPDarknet: 
# VAL集: acc_rate(90.83%) acc_map0.9_rate(94.19%) 
# 90,180,270 test集: acc_rate(22.56%) acc_map0.9_rate(36.59%) 
# config_file = 'work-dir/r3det-CSPDarknet/r3det-CSPDarknet_In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-CSPDarknet/epoch_40.pth'

# KFIoU-R3det-CSPDarknet: 
# VAL集: acc_rate(96.75%) acc_map0.9_rate(98.52%) 
# 90,180,270 test集: acc_rate(24.39%) acc_map0.9_rate(37.20%) 
# config_file = 'work-dir/r3det-kfiou-CSPDarknet/r3det_CSPDarknet_kfiou_In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-CSPDarknet/epoch_40.pth'

# KFIoU-R3det-Res18: 
# VAL集: acc_rate(98.67%) acc_map0.9_rate(99.70%) 
# NMS VAL集：acc_rate(95.86) acc_map0.9_rate(99.41%)
# 90,180,270 test集: acc_rate(43.90%) acc_map0.9_rate(59.76%) 
# config_file = 'work-dir/r3det-kfiou-resnet18/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18/epoch_40.pth'

# KFIoU-R3det-CSPDarknet: GPU(ms) CPU(ms) 训练数据为meanspadding + src + 90180270
# data/GangPi-random-rotate-meanspadding-src-90180270/test集：acc_rate(95.82%) acc_map0.9_rate(98.51%) 
# data/GangPi-random-rotate-meanspadding-src-90180270/test集 + NMS：acc_rate(94.93%) acc_map0.9_rate(98.81%)
config_file = 'work-dir/r3det-kfiou-resnet18-meanspadding-src-90180270/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
checkpoint_file = 'work-dir/r3det-kfiou-resnet18-meanspadding-src-90180270/epoch_40.pth'

# KFIoU-R3det-Res18-512:输入图像尺寸为512
# # VAL集: acc_rate(%) acc_map0.9_rate(%)
# 90,180,270 test集: acc_rate(21.34%) acc_map0.9_rate(34.14%) 
# config_file = 'work-dir/r3det-kfiou-resnet18-512/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18-512/epoch_40.pth'

#KFIoU-R3det-Res18-256: 输入图像尺寸为256
# # VAL集: acc_rate(80.03%) acc_map0.9_rate(90.98%)
# 90,180,270 test集: acc_rate(0.61%) acc_map0.9_rate(6.09%) 
# config_file = 'work-dir/r3det-kfiou-resnet18-256/res18_kfiou-In_r50_fpn_1x_GangPi_oc.py'
# checkpoint_file = 'work-dir/r3det-kfiou-resnet18-256/epoch_40.pth'

# build the model from a config file and a checkpoint file
#model = init_detector(config_file, checkpoint_file, device='cpu', cfg_options={'model.test_cfg.score_thr':0.5})
model = init_detector(config_file, checkpoint_file, device='cuda:3', cfg_options={'model.test_cfg.score_thr':0.5})
#model = init_detector(config_file, checkpoint_file, device='cuda:0', cfg_options={'model.test_cfg.score_thr':0.5})

# timers
_t = {'im_detect': Timer(), 'misc': Timer()}

# image-split中的val数据集中的前5张图片
# dir_path = '/home/zjhu/mm/mmrotate/data/rotate_map_test-5'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-5-res18'

#测试不区分类别的NMS以及离群筛除
# dir_path = '/home/zjhu/mm/mmrotate/data/NMS-test'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS_test-r3det-res18-kfiou'

# 整个image-split中的val数据集
# dir_path = '/home/zjhu/mm/mmrotate/data/rotate_map_test-676'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-CSPDarknet'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-CSPDarknet-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-res18-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS-rotate_map_test-676-r3det-res18-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-res18-512-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/rotate_map_test-676-r3det-res18-256-kfiou'

# 只有90，180，270三个角度的图像集164张
# dir_path = '/home/zjhu/mm/mmrotate/data/MultiAngle-map-test-164'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-CSPDarknet'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-CSPDarknet-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-res18-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS-MultiAngle-map-test-164-r3det-res18-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-res18-512-kfiou'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/MultiAngle-map-test-164-r3det-res18-256-kfiou'

dir_path = '/home/zjhu/mm/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270/test/images'
# save_path = '/home/zjhu/mm/mmrotate/show-dir/meanspadding-src-90180270-mAP-r3det-CSPDarknet-kfiou'
save_path = '/home/zjhu/mm/mmrotate/show-dir/NMS-meanspadding-src-90180270-mAP-r3det-CSPDarknet-kfiou'

if __name__ == '__main__':

    # 对各种模型进行识别准确率计算，通过计算每一张图片的AP，为1则说明整个图片识别正确
    # 最终计算完全正确的比例: acc_rate，和一张图中检测AP>0.9的比例: acc_ed1_rate
    # 测试集：经过image-split处理的val集，未经过任何处理的旋转（90，180，270）三个角度的测试集

    img_names = glob.glob(dir_path+'/*bmp')
    num_images = len(img_names)
    total_time = 0
    total = len(img_names)
    acc_count = 0 # 完全正确
    ed1_count = 0 # 错一个目标
    for i, img in enumerate(img_names):
        # test a single image and show the results
        #img = os.path.join(dir_path, img_name)  # or img = mmcv.imread(img), which will only load it once
        _t['im_detect'].tic()
        result = inference_detector(model, img)
        result_temp = result
        result_ori = copy.deepcopy(result)
    
        # ann_file = img.replace('bmp', 'txt')
        ann_file = img.replace('bmp', 'txt').replace('images', 'annfiles')
        annotation = load_ann(ann_file, 'oc')['ann']
        
        #对图像中可能存在的多个类别检测框检测出同一目标进行NMS后处理，只保留confidence最高的那个类别
        dets = np.vstack(result_temp)
        nms_dets = NMS(dets, thresh=0.3)
        print('dets: ', dets)

        temp = [-1,-1,-1,-1,-1]
        nms_dets.extend(temp)
        print("nms_dets: ", nms_dets)

        indx = 0
        all_del = 0 # 记录
        # 遍历10个类别的检测结果根据nms_dets进行删除
        for i in range(10):
            tag = 0
            print("len(result): ", len(result[i]))
            # 如果当前类别没有结果则跳过
            if len(result[i]) == 0:
                continue
            # 如果nms_dets中索引能和当前类别的检测框数目对应，则说明没有框被删除
            indx = indx + len(result[i])
            print("indx ", indx)

            if nms_dets[indx - 1] == indx - 1 + all_del:
                continue
            else:
                temp_indx = indx - len(result[i])
                for j in range(0, len(result[i])):
                    print("j: ", j)
                    if nms_dets[temp_indx + j] != temp_indx + j + all_del:
                        #print(type(result[i]))
                        result[i] = np.delete(result[i], j-tag, 0)
                        print(result[i])
                        indx = indx - 1
                        tag = tag + 1
                        all_del = all_del + 1 
                        print("tag: ", tag)

        print('img: ', img)
        print('result: ', result)
        # print('result_ori: ', result_ori)
        # print('ann: ', annotation)
        # print('cls_dets:', cls_dets)
        # print('cls_gts: ', cls_gts)
        

        # 根据一张图片中检测的map来判断整个图片的目标是否完全识别正确
        eval_result = evaluate([result], [annotation])
        print('eval_result: ', eval_result)
        if eval_result['mAP'] > 0.9:
            ed1_count = ed1_count + 1
            save_img_path = os.path.join(save_path, 'acc0.9-'+os.path.basename(img))
            if eval_result['mAP'] == 1.0:
                acc_count = acc_count + 1
                save_img_path = os.path.join(save_path, 'acc-'+os.path.basename(img))
        else:
            save_img_path = os.path.join(save_path, 'wrong-'+os.path.basename(img))
        detect_time = _t['im_detect'].toc(average=False)

        if i > 0:
            total_time += detect_time
        # or save the visualization results to image files
        model.show_result(img, result, out_file=save_img_path)
        # model.show_result(img, result_ori, out_file=save_img_path.replace('.bmp', '-ori.bmp'))
        # model.show_result(img, result, out_file=os.path.join(save_path, 'nms-'+os.path.basename(img)))

    print('acc_ed1_rate: ', ed1_count/total * 100, '%')
    print('acc_rate: ', acc_count/total * 100, '%')