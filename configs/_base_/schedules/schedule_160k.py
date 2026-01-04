# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)  # 学习率减小
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(policy='poly', power=0.9, min_lr=2e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=18000)
checkpoint_config = dict(by_epoch=False, interval=3000)
evaluation = dict(interval=3000, metric='mIoU', pre_eval=True)


