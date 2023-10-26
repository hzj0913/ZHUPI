
# Copyright (c) OpenMMLab. All rights reserved.
import torch
from turtle import forward
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from ..builder import ROTATED_HEADS
from .rotated_retina_head import RotatedRetinaHead
import numpy as np
import math

from mmrotate.core import (aug_multiclass_nms_rotated, bbox_mapping_back,
                           build_assigner, build_bbox_coder,
                           build_prior_generator, build_sampler,
                           multiclass_nms_rotated_landm, obb2hbb,
                           rotated_anchor_inside_flags)

# from ..builder import build_loss
from mmrotate.models import build_loss

# 能够额外检测关键点 + r3det实现360角度预测
@ROTATED_HEADS.register_module()
class Rotated360RetinaHead(RotatedRetinaHead):
    """
    The head contains three subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors the last one for the anchors landmarks

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 landm_coder=dict(
                     type='DeltaTLTRCECoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
                 loss_landm=dict(type='SmoothL1Loss', loss_weight=1.0),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        super(Rotated360RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

        self.loss_landm = build_loss(loss_landm)
        self.landm_coder = build_bbox_coder(landm_coder)
        # print('kwargs: ', kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        # 回归和分类的convs
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels # 256
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_bbox_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)
        self.retina_landm_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 6, 3, padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (torch.Tensor): Features of a single scale level.

        Returns:
            tuple (torch.Tensor):

                - cls_score (torch.Tensor): Cls scores for a single scale \
                    level the channels number is num_anchors * num_classes.
                - bbox_pred (torch.Tensor): Box energies / deltas for a \
                    single scale level, the channels number is num_anchors * 5.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        # print('x: ', x.shape)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_bbox_reg(reg_feat)
        landm_pred = self.retina_landm_reg(reg_feat)
        # print('bbox_pred: ', bbox_pred.shape)
        # print('landm_pred', landm_pred.shape)
        return cls_score, bbox_pred, landm_pred

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_landms,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            gt_landms (torch.Tensor): Ground truth landms of the image,
                shape (num_gts, 6).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - landms_targets_list (list[Tensor]): Landms targets of each level
                - landms_weights_list (list[Tensor]): Landms weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if self.assign_by_circumhbbox is not None: # oc
            gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_bboxes_assign, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            assign_result = self.assigner.assign(
                anchors, gt_bboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        # print('assgin_result: ',assign_result)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)
        # print('sampling_result: ', sampling_result)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        landm_targets = anchors.new_zeros((num_valid_anchors, 6), dtype=torch.float)
        landm_weights = anchors.new_zeros((num_valid_anchors, 6), dtype=torch.float)
        # print('box_targets.shape: ', bbox_targets.shape)
        # print('landm_targets.shape: ', landm_targets.shape)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        # print('labels.shape: ', labels.shape)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # print('neg_inds: ', len(neg_inds))
        pos_gt_landms = gt_landms[sampling_result.pos_assigned_gt_inds.long(), :]
        # print('sampling_result.pos_gt_bboxes: ', sampling_result.pos_gt_bboxes)
        # print('len sampling_result.pos_gt_bboxes: ', len(sampling_result.pos_gt_bboxes))
        # print('pos_gt_landms', pos_gt_landms)
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox: #False
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
                # print('posbox-shape: ', pos_bbox_targets.shape)
                pos_landms_targets = self.landm_coder.encode(
                    sampling_result.pos_bboxes, pos_gt_landms)
                # print('poslandms-shape: ', pos_landms_targets.shape)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_landms_targets = pos_gt_landms
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            landm_targets[pos_inds, :] = pos_landms_targets
            landm_weights[pos_inds, :] = 1.0
            # print('pos_inds:', len(pos_inds))
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            landm_targets = unmap(landm_targets, num_total_anchors, inside_flags)
            landm_weights = unmap(landm_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, landm_targets, landm_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_landms_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_landms_list (list[Tensor]): Ground truth landms of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - landms_targets_list (list[Tensor]): Landms targets of each level.
                - landms_weights_list (list[Tensor]): Landms weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_landms_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_landm_targets, all_landm_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:9]
        rest_results = list(results[9:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        # print('num_total_pos: ', num_total_pos) # 和pos_inds数目一致
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # print('num_total_neg: ', num_total_neg) # 和neg_inds数目一致
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        landm_targets_list = images_to_levels(all_landm_targets,
                                              num_level_anchors)
        landm_weights_list = images_to_levels(all_landm_weights,
                                              num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, landm_targets_list, landm_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'landm_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             landm_preds,
             gt_bboxes,
             gt_landms,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            landm_preds (list[Tensor]): Box landms for each scale
                level with shape (N, num_anchors * 6, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_landms (list[Tensor]): Ground truth landms for each image with
                shape (num_gts, 6) in [left_top_x, left_top_y, right_top_x, right_top_y, cx, cy]
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # print('img_metas: ', img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_landms,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, landm_targets_list,
         landm_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        # print("num_total_sample: ", num_total_samples)
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox, losses_landm = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            landm_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            landm_targets_list,
            landm_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_landm=losses_landm)
    
    def loss_single(self, cls_score, bbox_pred, landm_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, landm_targets, landm_weights, num_total_samples):

        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        # print('1111: ', cls_score.shape)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        # print('2222: ', cls_score.shape)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        # bbox loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        # diff_bbox = torch.abs(bbox_pred - bbox_targets)
        # diff_bbox = diff_bbox[torch.where(diff_bbox > 0.11)]
        # print('bbox_pred: ', bbox_pred)
        # print('bbox_targets: ', bbox_targets)
        # print('diff_bbox: ', diff_bbox)
        # print('bbox_target: ', bbox_targets)
        anchors = anchors.reshape(-1, 5)    
        # print('yesyesyes')
        bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)
        bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)

        # KFIoU loss
        # loss_bbox = self.loss_bbox(
        #     bbox_pred,
        #     bbox_targets,
        #     bbox_weights,
        #     pred_decode=bbox_pred_decode,
        #     targets_decode=bbox_targets_decode,
        #     avg_factor=num_total_samples)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        # print('loss_bbox: ', loss_bbox)


        # landm loss
        landm_targets = landm_targets.reshape(-1, 6)
        landm_weights = landm_weights.reshape(-1, 6)
        landm_pred = landm_pred.permute(0, 2, 3, 1).reshape(-1, 6)
        # print('landm_pred: ', landm_pred)
        # print('landm_targets: ', landm_targets)
        # diff_landm = torch.abs(landm_pred - landm_targets)
        # diff_landm = diff_landm[torch.where(diff_landm > 0.11)]
        # print('diff_landms: ', diff_landm)
        loss_landm = self.loss_landm(
            landm_pred,
            landm_targets,
            landm_weights,
            avg_factor=num_total_samples)
        # print('loss_landm: ', loss_landm)
        return loss_cls, loss_bbox, loss_landm    
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_landms=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_landms (Tensor): Ground truth landms of each box,
                shape (num_gts, 6).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_landms, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'landm_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   landm_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox and landmark predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            landm_preds (list[Tensor]): landm energies / deltas for each scale
                level with shape (N, num_anchors * 6, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds) == len(landm_preds)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            landm_pred_list = [
                landm_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    landm_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_list,
                                                    landm_pred_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           landm_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores for a single scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas for a single
                scale level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (cx, cy, w, h, a) and the
                6-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(landm_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_landms = []
        for cls_score, bbox_pred, landm_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, landm_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] == landm_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            landm_pred = landm_pred.permute(1, 2, 0).reshape(-1, 6)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                landm_pred = landm_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            landms = self.landm_coder.decode(
                anchors, landm_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_landms.append(landms)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_landms = torch.cat(mlvl_landms)
        # print('scale_factor: ', scale_factor)
        # print('new_scale: ', np.append(scale_factor, scale_factor[:2]))
        if rescale:
            # angle should not be rescaled
            mlvl_bboxes[:, :4] = mlvl_bboxes[:, :4] / mlvl_bboxes.new_tensor(
                scale_factor)
            mlvl_landms[:,:6] = mlvl_landms[:, :6] / mlvl_landms.new_tensor(
                np.append(scale_factor, scale_factor[:2]))
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # print('mlvl_bboxes: ', mlvl_bboxes.shape)
        # print('mlvl_landms: ', mlvl_landms.shape)
        if with_nms:
            # multiclass_nms_rotated_landm函数中将bbox和landm拼接起来 得到（N，12）结果，12 =  5 + 6 + 1
            det_results, det_labels = multiclass_nms_rotated_landm(
                mlvl_bboxes, mlvl_landms, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            # print('det_results: ', det_results.shape)
            return det_results, det_labels
        else:
            return mlvl_bboxes, mlvl_landms, mlvl_scores
        
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes(self, cls_scores, bbox_preds, landm_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind = best_ind.expand(-1, -1, -1, 5)

            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, self.num_anchors, 5)
            best_pred = bbox_pred.gather(
                dim=-2, index=best_ind).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)

            for img_id in range(num_imgs):
                best_ind_i = best_ind[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                best_bbox_i = self.bbox_coder.decode(best_anchor_i,
                                                     best_pred_i)
                # f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data-bbox.txt', 'a+')
                # f.write('best-bbox-' + str(best_bbox_i) + '\n')
                bboxes_list[img_id].append(best_bbox_i.detach())
                # print('best_bbox_i: ', best_bbox_i)

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def filter_bboxes_landm(self, cls_scores, bbox_preds, landm_preds):
        """Filter predicted bounding boxes at each position of the feature
        maps. Only one bounding boxes with highest score will be left at each
        position. This filter will be used in R3Det prior to the first feature
        refinement stage.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            landm_preds (list[Tensor]): Landmarks energies / deltas for each scale
                level with shape (N, num_anchors * 6, H, W)

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level \
                of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            cls_score = cls_scores[lvl]
            bbox_pred = bbox_preds[lvl]
            landm_pred = landm_preds[lvl]

            anchors = mlvl_anchors[lvl]

            cls_score = cls_score.permute(0, 2, 3, 1)
            cls_score = cls_score.reshape(num_imgs, -1, self.num_anchors,
                                          self.cls_out_channels)

            cls_score, _ = cls_score.max(dim=-1, keepdim=True)
            best_ind = cls_score.argmax(dim=-2, keepdim=True)
            best_ind_landm = best_ind.expand(-1, -1, -1, 6)
            best_ind_anchor = best_ind.expand(-1, -1, -1, 5)
            # print('best_ind: ', best_ind)
            # print('best_ind.shape: ', best_ind.shape)

            landm_pred = landm_pred.permute(0, 2, 3, 1)
            landm_pred = landm_pred.reshape(num_imgs, -1, self.num_anchors, 6)
            best_pred = landm_pred.gather(
                dim=-2, index=best_ind_landm).squeeze(dim=-2)

            anchors = anchors.reshape(-1, self.num_anchors, 5)
            print('num_imgs: ', num_imgs)
            for img_id in range(num_imgs):
                best_ind_i = best_ind_anchor[img_id]
                best_pred_i = best_pred[img_id]
                best_anchor_i = anchors.gather(
                    dim=-2, index=best_ind_i).squeeze(dim=-2)
                # print('best_pred_i: ', best_pred_i)
                # print('best_pred_i.shape: ', best_pred_i.shape)
                best_landm_i = self.landm_coder.decode(best_anchor_i,
                                                     best_pred_i)
                cx = best_landm_i[:, 4]
                cy = best_landm_i[:, 5]
                # 角度
                tl_x = best_landm_i[:, 0]
                tl_y = best_landm_i[:, 1]
                tr_x = best_landm_i[:, 2]
                tr_y = best_landm_i[:, 3]
                angle = torch.zeros_like(cx)
                dx = tr_x - tl_x
                # 图像的y坐标和原点是反的
                dy = tl_y - tr_y
                # print('len(dx): ', len(dx))
                # print('len(best_landm_i): ', len(best_landm_i))
                for j in range(len(dx)):
                    if math.atan2(dy[j], dx[j]) > 0:
                        angle[j] = math.atan2(dy[j], dx[j]) 
                    else:
                        angle[j] = (2*np.pi + math.atan2(dy[j], dx[j])) 
                # 此时角度是0-360的变成针对bbox的角度范围
                for i in range(len(best_landm_i)):
                    if angle[i] < np.pi/2:
                        angle[i] = -angle[i]
                    elif angle[i] < np.pi:
                        angle[i] = np.pi-angle[i]
                    elif angle[i] < 3*np.pi/2:
                        angle[i] = np.pi-angle[i]
                    elif angle[i] < 2*np.pi:
                        angle[i] = 2*np.pi-angle[i]
                # 宽高
                ctltr_x = (tl_x + tr_x)/2
                ctltr_y = (tl_y + tr_y)/2
                w = pow(pow(tl_x-tr_x,2)+pow(tl_y-tr_y,2),0.5)
                w = w.reshape(w.shape[0], 1)
                h = 2* pow(pow(ctltr_x-cx,2)+pow(ctltr_y-cy,2),0.5)
                cx = cx.reshape(cx.shape[0], 1)
                cy = cy.reshape(cy.shape[0], 1)
                angle = angle.reshape(angle.shape[0], 1)
                w = w.reshape(w.shape[0], 1)
                h = h.reshape(h.shape[0], 1)
                # print('cx: ', cx)
                # print('cx: ', cx.shape)
                # print('tl_x: ', tl_x)
                # print('tl_x: ', tl_x.shape)
                # print('tl_y: ', tl_y)
                # print('tl_y: ', tl_y.shape)
                # print('angle: ', angle)
                # print('angle: ', angle.shape)
                # print('w: ', w)
                # print('w: ', w.shape)
                best_bbox_i = torch.cat((cx, cy, w, h, angle), dim=1)
                f = open('/data1/hzj/mmrotate/work_dirs/work_dirs/360test-refine-final-flip-101-landmfliter-test/data-1.txt', 'a+')
                f.write('best-bbox: ')
                for i in best_bbox_i:
                    f.write(str(i) + '\n')
                print('best_bbox_i: ', best_bbox_i)
                print('best_bbox_i: ', best_bbox_i.shape)
                bboxes_list[img_id].append(best_bbox_i.detach())

        return bboxes_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds):
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
