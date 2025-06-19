"""
generate_ddim_imgs.py
 使用 DDIM 采样方式批量生成图像，并保存到本地目录。
"""

import os
import time
import math
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from denoising_diffusion_pytorch import Unet, GaussianDiffusion


# ------------------------------------------------------------------
#                         🧩 参数配置
# ------------------------------------------------------------------
ckpt_path       = '/home/featurize/work/denoising-diffusion-pytorch/results/model-25.pt'
device          = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir        = Path('ddim_imgs')

total_images    = 302       # ✅ 你可以在这里指定总共要生成多少张图片
batch_size      = 16         # 一次生成多少张，建议 <= 显存容量允许的最大值
image_size      = 128        # 图像尺寸（需与你训练时保持一致）

timesteps       = 1000       # 训练时使用的扩散步数
sampling_steps  = 50         # DDIM 采样步数（越少越快，通常 20~100）
ddim_eta        = 0.0        # DDIM 参数，设为 0 表示确定性采样
seed            = None         # 可设为 None 表示每次随机；设置数字表示可复现
# ------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str = 'cuda'):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_steps,
        ddim_sampling_eta=ddim_eta
    ).to(device)

    data = torch.load(checkpoint_path, map_location=device)
    state = data['model']
    to_load = {}
    for k,v in state.items():
        if k.startswith('model.'):
            new_k = k[len('model.'):]
            to_load[new_k] = v
    model.load_state_dict(to_load)
    model.eval()
    return diffusion


@torch.inference_mode()
def generate_and_save_images(diffusion, total_images, batch_size, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    pad_width = len(str(total_images))
    produced = 0
    t0 = time.time()

    print(f"🚀 开始 DDIM 采样，共需生成 {total_images} 张图片，每批 {batch_size} 张")

    pbar = tqdm(total=total_images, unit="img")
    while produced < total_images:
        cur_bs = min(batch_size, total_images - produced)
        shape = (cur_bs, 3, image_size, image_size)

        samples = diffusion.ddim_sample(shape, return_all_timesteps=False)
        samples = samples.clamp(0., 1.)

        for i in range(cur_bs):
            img_path = save_dir / f"{produced + i:0{pad_width}d}.png"
            save_image(samples[i], img_path)

        produced += cur_bs
        pbar.update(cur_bs)

    pbar.close()
    print(f"✅ 完成！{total_images} 张图像已保存至「{save_dir.resolve()}」，耗时 {time.time() - t0:.1f}s")


def main():
    if seed is not None:
        torch.manual_seed(seed)

    diffusion = load_model(ckpt_path, device)
    generate_and_save_images(diffusion, total_images, batch_size, save_dir)


if __name__ == '__main__':
    main()
