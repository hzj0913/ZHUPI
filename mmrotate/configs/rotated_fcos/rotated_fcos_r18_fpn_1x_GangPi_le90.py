_base_ = [
    '../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

dataset_type = 'DOTADataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
#            'R','S','T','U','V','W','X','Y','Z', )
# data_root = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/'
# data_root = '/data1/hzj/mmrotate/data/generate_data/'

data_root = '/data1/hzj/mmrotate/data/img_0418/'

angle_version = 'oc'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
    train=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))


model = dict(
    type='RotatedFCOS',
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
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='RotatedFCOSHead',
        num_classes=10,
        # num_classes=36,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        centerness_on_reg=True,
        separate_angle=False,
        scale_angle=True,
        bbox_coder=dict(
            type='DistanceAnglePointCoder', angle_version=angle_version),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.5,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# evaluation
evaluation = dict(interval=1, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
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