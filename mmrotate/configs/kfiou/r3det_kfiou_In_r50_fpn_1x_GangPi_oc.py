_base_ = ['../kfiou/r3det_kfiou_ln_r50_fpn_1x_dota_oc.py']

dataset_type = 'DOTADataset'
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', )
data_root = '/home/zjhu/mm/mmrotate/data/GangPI_randomAngle/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'train/images-split/annfiles/',
        img_prefix=data_root + 'train/images-split/images/'),
    val=dict(
        type=dataset_type,
        classes=classes, 
        ann_file=data_root + 'val/images-split/annfiles/',
        img_prefix=data_root + 'val/images-split/images/'),
    test=dict(
        type=dataset_type,
        classes=classes, 
        # ann_file=data_root + 'test/images-split/images/',
        # img_prefix=data_root + 'test/images-split/images/'))
        ann_file=data_root + 'test/images/',
        img_prefix=data_root + 'test/images/'))

angle_version = 'oc'
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='KFIoURRetinaHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=False,
            proj_xy=False,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0)),
    refine_heads=[
        dict(
            type='KFIoURRetinaRefineHead',
            num_classes=10,
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
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='KFLoss', fun='ln', loss_weight=5.0))
    ])

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

     
    
   
 



