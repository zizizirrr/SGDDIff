# =========================================================
# Diffusion Distillation Config for EORSSD
# Teacher + Student are BOTH diffusion models
# =========================================================

_base_ = [
    '../_base_/datasets/my_orsi4199.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

# ---------------------------------------------------------
# Teacher / Student config paths
# ---------------------------------------------------------
teacher_cfg = 'configs/ORSI-4199/orsi-4199_teacher.py'
student_cfg = 'configs/ORSI-4199/orsi-4199_student.py'

teacher_ckpt = 'work_dirs/ORSI-4199_teacher/best_mIoU_iter.pth'

# ---------------------------------------------------------
# Distillation Model
# ---------------------------------------------------------
model = dict(
    type='DiffusionDistiller',

    # =============== Teacher ==============================
    teacher=dict(
        cfg=teacher_cfg,
        pretrained=teacher_ckpt,
        frozen=True,                 # Teacher 不更新
        diffusion_steps=10,          # Teacher 扩散步数（完整）
    ),

    # =============== Student ==============================
    student=dict(
        cfg=student_cfg,
        diffusion_steps=3,           # Student 扩散步数（轻量）
    ),

    # =============== Distillation =========================
    distill_cfg=dict(
        # ① 特征蒸馏
        feature_kd=True,
        feature_layers=['decode_head.layer4'],  # 第4层对齐
        feature_loss=dict(
            type='L2Loss',
            loss_weight=1.0,
        ),

        # ② 扩散轨迹蒸馏
        diffusion_kd=True,
        trajectory_loss=dict(
            type='MSELoss',
            loss_weight=0.5,
        ),

        # ③ logits 蒸馏（可选）
        logit_kd=True,
        logit_loss=dict(
            type='KLDivergence',
            temperature=4.0,
            loss_weight=0.5,
        ),
    ),
)

# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
)

# ---------------------------------------------------------
# Optimizer (只优化 Student)
# ---------------------------------------------------------
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=6e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

# ---------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
)

# ---------------------------------------------------------
# Runtime
# ---------------------------------------------------------
find_unused_parameters = True
evaluation = dict(
    interval=10000,
    metric='mIoU',
    save_best='mIoU'
)
