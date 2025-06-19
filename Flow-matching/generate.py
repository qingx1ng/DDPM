import os
import torch
import numpy as np
from unet import Unet
from flow import OptimalTransportFlow, sample_images
from utils import make_im_grid, load_checkpoint

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

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型配置（必须与你训练时一致）
    model = Unet(ch=256, att_channels=[0, 1, 1, 0], dropout=0.0).to(device)
    # model = torch.compile(model)

    # EMA model 用于生成更清晰图像
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )

    # 加载权重
    checkpoint_path = '/home/featurize/work/Flow-matching/ckp_epoch50_step31875.tar'  # 替换为你的模型文件路径
    # ckpt = torch.load(checkpoint_path, map_location='cpu')  # 使用 CPU 加载更安全
    # # 打印所有顶层键
    # print("Top-level keys in checkpoint:")
    # print(ckpt['model_state_dict'].keys())
    # print('-' * 50)
    step, epoch, model, _, _, ema_model = load_checkpoint(
        checkpoint_path, model, None, None, ema_model
    )

    print(f'Loaded model from epoch {epoch}, step {step}')

    # 设置评估模式
    model.eval()
    ema_model.eval()

    os.makedirs('generate_samples', exist_ok=True)

    with torch.no_grad():
        shape = (8, 3, 128, 128)  # 与训练保持一致
        num_steps = 2           # 与 sample_images 中一致

        print('Generating images from model...')
        x = sample_images(model, shape, num_steps = 2)
        x = x[-1]
        grid = make_im_grid(x, (2, 4))
        grid.save('generate_samples/sample.png')

        print('Generating images from EMA model...')
        x_ema = sample_images(ema_model, shape, num_steps = 2)
        x_ema = x_ema[-1]
        grid_ema = make_im_grid(x_ema, (2, 4))
        grid_ema.save('generate_samples/sample_ema.png')

    print('Generation complete.')
