"""
    任务2 训练ddpm，参数设置如下
"""

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
) 

trainer = Trainer(
    diffusion,
    '/home/featurize/work/LSNU/train_5000',
    train_batch_size = 32,
    save_and_sample_every = 7000,  # 每1000步保存一次生成样本
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False              # whether to calculate fid during training
)

trainer.train()