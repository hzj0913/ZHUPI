# backbone使用resnet18 

# 使用KFIoU 和角点回归

_base_ = ['../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_40e.py',
    '../_base_/default_runtime.py']

dataset_type = 'DOTALandmDataset'
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', )

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )

# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
#            'R','S','T','U','V','W','X','Y','Z', )

# data_root = '/home/zjhu/mm/mmrotate/data/GangPI_randomAngle/'
# data_root = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/'

# 钢厂新版本数据 彩色
# data_root = '/data1/hzj/mmrotate/data/img_cut/'
# data_root = '/data1/hzj/mmrotate/data/img_pre_my/'
# 彩色实验用数据集（经过随机角度增强）
# data_root = '/data1/hzj/mmrotate/data/img_0418/'

# 钢厂换摄像头后的数据
data_root = '/data1/hzj/mmrotate/data/img_0526/'


# data_root = '/data1/hzj/mmrotate/data/GangPi_dataloader_test-1/'
# data_root = '/data1/hzj/mmrotate/data/GangPi_dataloader_test-50/'
# data_root = '/data1/hzj/mmrotate/data/GangPi_dataloader_test-51/'

# 生成的数据集
# data_root = '/data1/hzj/mmrotate/data/generate_data/'


angle_version = '360'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationsLandms', with_bbox=True, with_landms=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    # dict(type='RResize', img_scale=(512, 512)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        # flip_ratio = [0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        # direction=['diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle_'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_landms', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        # img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # samples_per_gpu=1,
    # workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes, 
        # ann_file=data_root + 'train/annfiles/',
        # ann_file=data_root + 'train/annfiles_new/',
        # img_prefix=data_root + 'train/images/',
        # ann_file=data_root + 'train/annfiles/',
        # img_prefix=data_root + 'train/images/',
        ann_file=data_root + 'all/annfiles/',
        img_prefix=data_root + 'all/images/',
        pipeline=train_pipeline,
        version = angle_version),
    val=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'all/annfiles/',
        img_prefix=data_root + 'all/images/',
        # ann_file=data_root + 'train/annfiles/',
        # img_prefix=data_root + 'train/images/',
        # ann_file=data_root + 'val/annfiles/',
        # img_prefix=data_root + 'val/images/',
        # ann_file=data_root + 'val/annfiles/',
        # ann_file=data_root + 'val/annfiles_new/',
        # img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline,
        version = angle_version),
    test=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/',
        # ann_file=data_root + 'test/annfiles/',
        # img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        version = angle_version))

model = dict(
    type='RALandm',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='Rotated360RetinaHead',
        num_classes=10,
        # num_classes=11,
        # num_classes=36,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox='oc',
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range='oc',
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        landm_coder=dict(
            type='DeltaTLTRCECoder',
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0),
        # loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=1.0),
        loss_landm=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
        # loss_landm=dict(type='SmoothL1Loss', beta=0.11, loss_weight=4.0)),
    frm_cfgs=[dict(in_channels=256, featmap_strides=[8, 16, 32, 64, 128])],
    num_refine_stages=1,
    refine_heads=[
        dict(
            type='Rotated360RetinaRefineHead',
            num_classes=10,
            # num_classes=11,
            # num_classes=36,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            assign_by_circumhbbox=None,
            anchor_generator=dict(
                type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=False,
                proj_xy=False,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            landm_coder=dict(
                type='DeltaTLTRCECoder',
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0),
            # loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=1.0),
            loss_landm=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0))
            # loss_landm=dict(type='SmoothL1Loss', beta=0.11, loss_weight=4.0))
    ],
    train_cfg=dict(
        s0=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        sr=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.5,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

load_from = '/data2/hzj/mmrotate/work_dirs/images_0418/latest.pth'

# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=1)

     
    
   
 



