# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.008  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(policy='step', warmup=None, step=[24, 32])
# # runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=36)


#lr_config = dict(policy='CosineRestart', warmup=None, periods=[24,4,4,4,4,4], min_lr=0, restart_weights=[1,1,1,1,1,1])
#lr_config = dict(policy='CosineRestart', warmup=None, periods=[12,8,8,8,8,8], min_lr=0, restart_weights=[1,1,1,1,1,1])#2
#lr_config = dict(policy='CosineRestart', warmup=None, periods=[18,6,6,6,6,10], min_lr=0, restart_weights=[1,1,1,1,1,1])#3
#lr_config = dict(policy='CosineRestart', warmup=None, periods=[24,12,4 ,12], min_lr=0, restart_weights=[1,1,1,1])#4
lr_config = dict(policy='CosineRestart', warmup=None, periods=[12,12,12,4,12,12], min_lr=0, restart_weights=[1,1,1,1,1,1])#5
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=64)




# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# lr_config = dict(
#     policy="CosineAnnealing",
#     min_lr=0
# )
# # runtime settings
# total_epochs = 44