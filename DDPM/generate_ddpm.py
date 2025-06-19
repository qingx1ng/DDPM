"""
    ddpm生成和测试集一样数量的图像
"""
import math, os, time
import torch
from torchvision.utils import save_image
from pathlib import Path
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# ------------------------------------------------------------------
#                         🧩 参数配置
# ------------------------------------------------------------------
ckpt_path = '/home/featurize/work/denoising-diffusion-pytorch/results/model-30.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
total_images = 302               # 总共要生成多少张图像
batch_size = 16                    # 每批生成多少张
save_dir = Path("ddpm_imgs")       # 保存目录
image_size = 128                   # 生成图像的分辨率（必须与你训练时一致）
sampling_timesteps = 250          # 采样步数（越少越快）
timesteps = 1000                  # 训练时设置的总扩散步数

# ======== 加载模型 ========
def load_model(ckpt_path, device):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=timesteps,
        sampling_timesteps=sampling_timesteps
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)['model']
    to_load = {}
    for k,v in state.items():
        if k.startswith('model.'):
            new_k = k[len('model.'):]
            to_load[new_k] = v
    model.load_state_dict(to_load)
    model.eval()
    return diffusion

# ======== 批量生成图像并保存 ========
@torch.inference_mode()
def generate_images(diffusion, total_images, batch_size, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    pad = len(str(total_images))
    idx = 0

    print(f"开始生成 {total_images} 张图像，保存到 {save_dir.resolve()}")
    t0 = time.time()

    while idx < total_images:
        cur_bs = min(batch_size, total_images - idx)
        samples = diffusion.sample(batch_size=cur_bs)
        samples = samples.clamp(0., 1.)

        for i in range(cur_bs):
            img_path = save_dir / f"{idx + i:0{pad}d}.png"
            save_image(samples[i], img_path)

        idx += cur_bs
        print(f"  -> 已生成 {idx}/{total_images} 张", end='\r', flush=True)

    print(f"\n✅ 生成完成！耗时 {time.time() - t0:.1f}s")

# ======== 主程序入口 ========
if __name__ == '__main__':
    diffusion = load_model(ckpt_path, device)
    generate_images(diffusion, total_images, batch_size, save_dir)
