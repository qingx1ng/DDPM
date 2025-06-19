import os
import numpy as np
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.cuda.amp import GradScaler

from unet import Unet
from flow import OptimalTransportFlow, sample_images
from utils import get_loaders, make_im_grid, make_checkpoint


# 设置随机种子
torch.manual_seed(159753)
np.random.seed(159753)

# CUDA 性能设置
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


# 构建 loss 函数
def get_loss_fn(model: Unet, flow: OptimalTransportFlow):
    def loss_fn(batch: Tensor) -> Tensor:
        t = torch.rand(batch.shape[0], device=batch.device)
        x0 = torch.randn_like(batch)
        xt = flow.step(t, x0, batch)
        pred_vel = model(xt, t)
        true_vel = flow.target(t, x0, batch)
        return MSELoss()(pred_vel, true_vel)
    return loss_fn


# 动态学习率策略
def get_lr(config, step):
    if step < config['warmup_steps']:
        return config['min_lr'] + (config['max_lr'] - config['min_lr']) * (step / config['warmup_steps'])
    if step > config['max_steps']:
        return config['min_lr']
    decay_ratio = (step - config['warmup_steps']) / (config['max_steps'] - config['warmup_steps'])
    return config['max_lr'] - (config['max_lr'] - config['min_lr']) * decay_ratio


if __name__ == '__main__':
    os.makedirs('samples', exist_ok=True)

    config = {
        'train_dir': '/home/featurize/work/LSNU/train_5000',  # 替换为你的训练集路径
        'test_dir': '/home/featurize/work/LSNU/test',          # 替换为你的测试集路径
        'sigma_min': 3e-2,
        'min_lr': 1e-8,
        'max_lr': 2e-4,
        'warmup_steps': 45000,
        'epochs': 2000,
        'max_steps': 400000,
        'batch_size': 4,
        'log_freq': 100,
        'num_workers': 2,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型与优化器
    model = Unet(ch=256, att_channels=[0, 1, 1, 0], dropout=0.0).to(device)
    # model = torch.compile(model)
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )

    flow = OptimalTransportFlow(config['sigma_min'])
    loss_fn = get_loss_fn(model, flow)
    optim = torch.optim.Adam(model.parameters(), lr=config['min_lr'])

    train_loader, _ = get_loaders(config)
    scaler = GradScaler()

    step = 0
    curr_epoch = 0
    accumulation_steps = 2

    for epoch in range(curr_epoch, config['epochs'] + 1):
        model.train()
        ema_model.train()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)

            if i % accumulation_steps == 0:
                optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device):
                loss = loss_fn(x) / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optim)
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optim)
                scaler.update()
                ema_model.update_parameters(model)

                for g in optim.param_groups:
                    g['lr'] = get_lr(config, step)

                if (step + 1) % config['log_freq'] == 0:
                    print(f'Step: {step} ({epoch}) | Loss: {loss.item() * accumulation_steps:.5f} | Grad: {grad.item():.5f} | Lr: {g["lr"]:.3e}')
                step += 1

        # 每 10 个 epoch 验证一次采样 + 每 40 个 epoch 存 checkpoint
        if epoch % 30 == 0 or epoch == config['epochs']:
            model.eval()
            ema_model.eval()
            with torch.no_grad():
                print(f'[Eval] Generating samples at epoch {epoch}')
                shape = (8, 3, 128, 128)
                gen_x = sample_images(model, shape, num_steps=25)[-1]
                gen_x_ema = sample_images(ema_model, shape, num_steps=25)[-1]

                image = make_im_grid(gen_x, (2, 4))
                image.save(f'samples/{epoch}.png')

                image_ema = make_im_grid(gen_x_ema, (2, 4))
                image_ema.save(f'samples/ema_{epoch}.png')

        if epoch % 5 == 0:
                make_checkpoint(f'ckp_epoch{epoch}_step{step}.tar', step, epoch, model, optim, scaler, ema_model)

    # 最后保存最终模型
    make_checkpoint(f'ckp_final_step{step}.tar', step, config['epochs'], model, optim, scaler, ema_model)
