_base_ = [
    '../_base_/datasets/my_eorssd.py',
    '../_base_/default_runtime.py',
]

# ======================
# Teacher Model
# ======================
teacher = dict(
    type='SGD',
    timesteps=10,
    randsteps=5,
    accumulation=True,
    bit_scale=0.01,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        # init_cfg=None,
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='MultiStageMerging',
            in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=None)
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ======================
# Student Model
# ======================
student = dict(
    type='SGD',
    timesteps=3,
    randsteps=1,
    accumulation=True,
    bit_scale=0.01,
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        # init_cfg=None,
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg),
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        dict(
            type='MultiStageMerging',
            in_channels=[256, 256, 256, 256],
            out_channels=256,
            kernel_size=1,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=None)
    ],
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4)),
    decode_head=dict(
        type='DeformableHeadWithTime',
        in_channels=[256],
        channels=256,
        in_index=[0],
        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        num_feature_levels=1,
        encoder=dict(
            type='DetrTransformerEncoder',
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                use_time_mlp=True,
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256,
                    num_levels=1,
                    num_heads=8,
                    dropout=0.),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    ffn_drop=0.,
                    act_cfg=dict(type='GELU')),
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ======================
# Teacher Checkpoint
# ======================
teacher_ckpt = 'checkpoint/EORSSD.pth'

# ======================
# Distillation Hyper-Params
# ======================
distill_cfg = dict(
    lambda_seg=1.0,
    lambda_kd=1.0,
    lambda_traj=1.0,
    temperature=4.0,
)

# ======================
# Optimizer (Student only)
# ======================
optimizer = dict(
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01
)

# ======================
# Training Length
# ======================
total_iters = 160000

work_dir = 'work_dirs/sgdd_distill'
