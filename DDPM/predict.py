"""
    ddpm采样生成图像脚本
"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image

ckpt_path = '/home/featurize/work/denoising-diffusion-pytorch/results/model-22.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4

def load_model(checkpoint_path, device='cuda'):
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = False
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,
        sampling_timesteps = 250
    ).to(device)

    # 加载权重
    data = torch.load(checkpoint_path, map_location=device)
    state = data['model']
    to_load = {}
    for k,v in state.items():
        if k.startswith('model.'):
            new_k = k[len('model.'):]
            to_load[new_k] = v
    model.load_state_dict(to_load)
    model.to(device).eval()

    return diffusion  # 这里返回 diffusion

def generate_and_save(diffusion, batch_size=4):
    with torch.no_grad():
        samples = diffusion.sample(batch_size = batch_size)
        samples = samples.clamp(0., 1.)
        samples = (samples * 255).type(torch.uint8).cpu()

    save_image(samples.float() / 255., 'sample_grid.png', nrow=int(batch_size ** 0.5))
    for i, img in enumerate(samples):
        save_image(img.float() / 255., f'sample_{i}.png')

if __name__ == '__main__':
    diffusion = load_model(ckpt_path, device)  # 接收返回值
    generate_and_save(diffusion, batch_size)
    print("✅ 生成完成：sample_grid.png 和 sample_*.png")
