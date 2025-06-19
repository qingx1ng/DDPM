"""
    任务4 在DDIM采样过程中，得到每一步中对x_0的预测
"""
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image
from tqdm import tqdm
import os

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
        sampling_timesteps = 50,  # DDIM采样步数通常小于1000，可调节加速
        ddim_sampling_eta = 0.0   # eta=0时为确定性DDIM采样
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

def generate_and_save_ddim(diffusion, batch_size=4):
    shape = (batch_size, 3, 128, 128)
    output_dir = 'results_ddim'

    # 如果文件夹不存在就创建
    os.makedirs(output_dir, exist_ok=True)

    imgs_all, x0_preds = diffusion.ddim_sample(shape, return_all_timesteps=True)

    final_imgs = imgs_all[:, -1]
    save_image(final_imgs.clamp(0,1), os.path.join(output_dir, 'ddim_sample_grid.png'), nrow=int(batch_size ** 0.5))

    for step in range(x0_preds.shape[0]):
        step_imgs = x0_preds[step].clamp(0,1)
        save_image(step_imgs, os.path.join(output_dir, f'x0_pred_step_{step}.png'), nrow=int(batch_size ** 0.5))

    print(f"✅ DDIM采样完成，结果保存在 {output_dir} 文件夹下。")

if __name__ == '__main__':
    diffusion = load_model(ckpt_path, device)
    generate_and_save_ddim(diffusion, batch_size)
