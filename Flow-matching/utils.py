"""
    修改utils的权重加载方式，训练好的权重有不匹配的key
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from einops import rearrange


class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f)
                            for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # dummy label


unloader = v2.Compose([
    v2.Lambda(lambda t: (t + 1) * 0.5),
    v2.Lambda(lambda t: t.permute(0, 2, 3, 1)),
    v2.Lambda(lambda t: t * 255.)
])


def make_im_grid(x0: torch.Tensor, xy: tuple = (1, 10)):
    x, y = xy
    im = unloader(x0.cpu())
    B, C, H, W = x0.shape
    im = rearrange(im, '(x y) h w c -> (x h) (y w) c', x=B // x, y=B // y).numpy().astype(np.uint8)
    im = v2.ToPILImage()(im)
    return im


def get_loaders(config):
    train_dir = config['train_dir']
    test_dir = config['test_dir']

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = UnlabeledImageDataset(train_dir, transform=transform)
    test_dataset = UnlabeledImageDataset(test_dir, transform=transform)

    bs = config['batch_size']
    j = config['num_workers']

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=j, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=j, pin_memory=True, drop_last=True)

    return train_loader, test_loader


def make_checkpoint(path, step, epoch, model, optim=None, scaler=None, ema_model=None):
    checkpoint = {
        'epoch': int(epoch),
        'step': int(step),
        'model_state_dict': model.state_dict(),
    }

    if optim is not None:
        checkpoint['optim_state_dict'] = optim.state_dict()

    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()

    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path, model, optim=None, scaler=None, ema_model=None):
    checkpoint = torch.load(path, weights_only=True)
    step = int(checkpoint['step'])
    epoch = int(checkpoint['epoch'])
    def strip_orig_mod(state_dict):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # 加载模型权重
    clean_model_state = strip_orig_mod(checkpoint['model_state_dict'])
    model.load_state_dict(clean_model_state, strict=False)
    model.eval()

    if optim is not None:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    if ema_model is not None:
        clean_model_state1 = strip_orig_mod(checkpoint['ema_model_state_dict'])
        ema_model.load_state_dict(clean_model_state1, strict=False)
        ema_model.eval()

    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    model.eval()
    return step, epoch, model, optim, scaler, ema_model
