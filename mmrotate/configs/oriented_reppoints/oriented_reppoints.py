_base_ = ['./oriented_reppoints_r50_fpn_1x_dota_le135.py']

dataset_type = 'DOTADataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
#            'R','S','T','U','V','W','X','Y','Z', )

# data_root = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/'
data_root = '/data1/hzj/mmrotate/data/img_0418/'
# 生成的数据集
# data_root = '/data1/hzj/mmrotate/data/generate_data/'


angle_version = 'le135'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RResize',
        img_scale=[(1333, 768), (1333, 1280)],
        multiscale_mode='range'),
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

model = dict(bbox_head=dict(num_classes=10))

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
        img_prefix=data_root + 'val/images/'),
    test=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'test/annfiles/',
        img_prefix=data_root + 'test/images/'))

# evaluation
evaluation = dict(interval=2, metric='mAP')
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[24, 32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)
checkpoint_config = dict(interval=2)