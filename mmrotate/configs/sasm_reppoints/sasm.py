_base_ = ['../rotated_reppoints/rotated_reppoints_r50_fpn_1x_dota_oc.py']

dataset_type = 'DOTADataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
# classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q',
#            'R','S','T','U','V','W','X','Y','Z', )

data_root = '/data1/hzj/mmrotate/data/GangPi-random-rotate-meanspadding-src-90180270-new/'
# data_root = '/data1/hzj/mmrotate/data/img_0418/'
# 生成的数据集
# data_root = '/data1/hzj/mmrotate/data/generate_data/'

model = dict(
    bbox_head=dict(
        type='SAMRepPointsHead',
        num_classes=10,
        loss_bbox_init=dict(type='BCConvexGIoULoss', loss_weight=0.375)),

    # training and testing settings
    train_cfg=dict(
        refine=dict(
            _delete_=True,
            assigner=dict(type='SASAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/'),
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