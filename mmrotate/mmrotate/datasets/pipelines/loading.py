# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile
from mmdet.datasets.pipelines import LoadAnnotations

from ..builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class LoadAnnotationsLandms(LoadAnnotations):
    """Load multiple types of annotations."""
    # 相较于原始的加载标签的函数，该函数额外对landms（左上，右上，中心点）进行操作。
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_landms=True, 
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 denorm_bbox=False,
                 file_client_args=dict(backend='disk')):
        super(LoadAnnotationsLandms, self).__init__(with_bbox, with_label, with_mask, with_seg, poly2mask, 
                                                    denorm_bbox, file_client_args)
        self.with_lanmds = with_landms

    
    def _load_landms(self, results):
        results['gt_landms'] = results['ann_info']['landms'].copy()
        results['landm_fields'].append('gt_landms')
        # print('resuls_loading: ', results)
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_lanmds:
            results = self._load_landms(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_landms={self.with_lanmds}'
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str

@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results
