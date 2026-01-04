_base_ = [
    '../_base_/datasets/my_eorssd.py',
    '../_base_/default_runtime.py',
]

# ======================
# Teacher Model
# ======================
teacher = dict(
    type='DDP',
    timesteps=50,
    randsteps=5,
    accumulation=True,
    bit_scale=0.01,

    backbone=...,        # ← 直接复制你原来的
    neck=...,
    decode_head=...,
    auxiliary_head=...,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ======================
# Student Model
# ======================
student = dict(
    type='DDP',
    timesteps=3,
    randsteps=1,
    accumulation=True,
    bit_scale=0.01,

    backbone=...,        # ← 和 teacher 完全一致
    neck=...,
    decode_head=...,
    auxiliary_head=...,
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# ======================
# Teacher Checkpoint
# ======================
teacher_ckpt = 'segmentation/checkpoint/EORSSD.pth'

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

work_dir = 'work_dirs/ddp_distill'
