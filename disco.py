import gc
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from functools import partial
from glob import glob
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from IPython import display
from PIL import Image, ImageOps
from ipywidgets import Output
from rich.progress import track
from torch import nn
from torch.nn import functional as F


def gitclone(url):
    res = subprocess.run(['git', 'clone', url], stdout=subprocess.PIPE).stdout.decode(
        'utf-8'
    )
    print(res)


def wget(url, outputdir):
    res = subprocess.run(
        ['wget', url, '-P', f'{outputdir}'], stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    print(res)


root_path = os.getcwd()


def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)


initDirPath = f'{root_path}/init_images'
createPath(initDirPath)
outDirPath = f'{root_path}/images_out'
createPath(outDirPath)

model_path = f'{root_path}/models'
createPath(model_path)

useCPU = False

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

PROJECT_DIR = os.path.abspath(os.getcwd())

root_path = os.getcwd()
model_path = f'{root_path}/models'

multipip_res = subprocess.run(
    [
        'pip',
        'install',
        'lpips',
        'datetime',
        'timm',
        'ftfy',
        'einops',
        'pytorch-lightning',
        'omegaconf',
    ],
    stdout=subprocess.PIPE,
).stdout.decode('utf-8')
print(multipip_res)

try:
    from CLIP import clip
except:
    if not os.path.exists("CLIP"):
        gitclone("https://github.com/openai/CLIP")
    sys.path.append(f'{PROJECT_DIR}/CLIP')

try:
    from guided_diffusion.script_util import create_model_and_diffusion
except:
    if not os.path.exists("guided-diffusion"):
        gitclone("https://github.com/crowsonkb/guided-diffusion")
    sys.path.append(f'{PROJECT_DIR}/guided-diffusion')

try:
    from resize_right import resize
except:
    if not os.path.exists("ResizeRight"):
        gitclone("https://github.com/assafshocher/ResizeRight.git")
    sys.path.append(f'{PROJECT_DIR}/ResizeRight')

import lpips

from CLIP import clip
from resize_right import resize
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

os.chdir(f'{PROJECT_DIR}')

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not useCPU) else 'cpu')
print('Using device:', DEVICE)
device = DEVICE  # At least one of the modules expects this name..


# https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869


def interp(t):
    return 3 * t ** 2 - 2 * t ** 3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(octaves, width, height, grayscale, device=device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    # out_array = [0.0] if grayscale else [0.0, 0.0, 0.0]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True):
    out = perlin_ms(octaves, width, height, grayscale)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin():
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, False)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, True)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)
    else:
        init = create_perlin_noise([1.5 ** -i * 0.5 for i in range(12)], 1, 1, False)
        init2 = create_perlin_noise([1.5 ** -i * 0.5 for i in range(8)], 4, 4, True)

    init = (
        TF.to_tensor(init)
        .add(TF.to_tensor(init2))
        .div(2)
        .to(device)
        .unsqueeze(0)
        .mul(2)
        .sub(1)
    )
    del init2
    return init.expand(batch_size, -1, -1, -1)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith(
            'https://'
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])


def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


padargs = {}


class MakeCutoutsDango(nn.Module):
    def __init__(
            self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def do_run():
    seed = args.seed
    print(range(args.start_frame, args.max_frames))

    for frame_num in range(args.start_frame, args.max_frames):

        display.clear_output(wait=True)

        if args.init_image in ['', 'none', 'None', 'NONE']:
            init_image = None
        else:
            init_image = args.init_image
        skip_steps = args.skip_steps

        loss_values = []

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        weights = []

        if args.prompts_series is not None and frame_num >= len(args.prompts_series):
            frame_prompt = args.prompts_series[-1]
        elif args.prompts_series is not None:
            frame_prompt = args.prompts_series[frame_num]
        else:
            frame_prompt = []

        print(args.image_prompts_series)
        if args.image_prompts_series is not None and frame_num >= len(
                args.image_prompts_series
        ):
            image_prompt = args.image_prompts_series[-1]
        elif args.image_prompts_series is not None:
            image_prompt = args.image_prompts_series[frame_num]
        else:
            image_prompt = []

        print(f'Frame {frame_num} Prompt: {frame_prompt}')

        model_stats = []
        for clip_model in clip_models:
            cutn = 16
            model_stat = {
                "clip_model": None,
                "target_embeds": [],
                "make_cutouts": None,
                "weights": [],
            }
            model_stat["clip_model"] = clip_model

            for prompt in frame_prompt:
                txt, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

                if args.fuzzy_prompt:
                    for i in range(25):
                        model_stat["target_embeds"].append(
                            (txt + torch.randn(txt.shape).cuda() * args.rand_mag).clamp(
                                0, 1
                            )
                        )
                        model_stat["weights"].append(weight)
                else:
                    model_stat["target_embeds"].append(txt)
                    model_stat["weights"].append(weight)

            if image_prompt:
                model_stat["make_cutouts"] = MakeCutouts(
                    clip_model.visual.input_resolution, cutn, skip_augs=skip_augs
                )
                for prompt in image_prompt:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(fetch(path)).convert('RGB')
                    img = TF.resize(
                        img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS
                    )
                    batch = model_stat["make_cutouts"](
                        TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1)
                    )
                    embed = clip_model.encode_image(normalize(batch)).float()
                    if fuzzy_prompt:
                        for i in range(25):
                            model_stat["target_embeds"].append(
                                (
                                        embed + torch.randn(embed.shape).cuda() * rand_mag
                                ).clamp(0, 1)
                            )
                            weights.extend([weight / cutn] * cutn)
                    else:
                        model_stat["target_embeds"].append(embed)
                        model_stat["weights"].extend([weight / cutn] * cutn)

            model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
            model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
            if model_stat["weights"].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            model_stat["weights"] /= model_stat["weights"].sum().abs()
            model_stats.append(model_stat)

        init = None
        if init_image is not None:
            init = Image.open(fetch(init_image)).convert('RGB')
            init = init.resize((args.side_x, args.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)

        if args.perlin_init:
            if args.perlin_mode == 'color':
                init = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
                )
                init2 = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(8)], 4, 4, False
                )
            elif args.perlin_mode == 'gray':
                init = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(12)], 1, 1, True
                )
                init2 = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(8)], 4, 4, True
                )
            else:
                init = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(12)], 1, 1, False
                )
                init2 = create_perlin_noise(
                    [1.5 ** -i * 0.5 for i in range(8)], 4, 4, True
                )
            # init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device)
            init = (
                TF.to_tensor(init)
                .add(TF.to_tensor(init2))
                .div(2)
                .to(device)
                .unsqueeze(0)
                .mul(2)
                .sub(1)
            )
            del init2

        cur_t = None

        def cond_fn(x, t, y=None):
            with torch.enable_grad():
                x_is_NaN = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if use_secondary_model is True:
                    alpha = torch.tensor(
                        diffusion.sqrt_alphas_cumprod[cur_t],
                        device=device,
                        dtype=torch.float32,
                    )
                    sigma = torch.tensor(
                        diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                        device=device,
                        dtype=torch.float32,
                    )
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    out = secondary_model(x, cosine_t[None].repeat([n])).pred
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                    out = diffusion.p_mean_variance(
                        model, x, my_t, clip_denoised=False, model_kwargs={'y': y}
                    )
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                    x_in = out['pred_xstart'] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                for model_stat in model_stats:
                    for i in range(args.cutn_batches):
                        t_int = (
                                int(t.item()) + 1
                        )  # errors on last step without +1, need to find source
                        # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                        try:
                            input_resolution = model_stat[
                                "clip_model"
                            ].visual.input_resolution
                        except:
                            input_resolution = 224

                        cuts = MakeCutoutsDango(
                            input_resolution,
                            Overview=args.cut_overview[1000 - t_int],
                            InnerCrop=args.cut_innercut[1000 - t_int],
                            IC_Size_Pow=args.cut_ic_pow,
                            IC_Grey_P=args.cut_icgray_p[1000 - t_int],
                        )
                        clip_in = normalize(cuts(x_in.add(1).div(2)))
                        image_embeds = (
                            model_stat["clip_model"].encode_image(clip_in).float()
                        )
                        dists = spherical_dist_loss(
                            image_embeds.unsqueeze(1),
                            model_stat["target_embeds"].unsqueeze(0),
                        )
                        dists = dists.view(
                            [
                                args.cut_overview[1000 - t_int]
                                + args.cut_innercut[1000 - t_int],
                                n,
                                -1,
                            ]
                        )
                        losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                        loss_values.append(
                            losses.sum().item()
                        )  # log loss, probably shouldn't do per cutn_batch
                        x_in_grad += (
                                torch.autograd.grad(
                                    losses.sum() * clip_guidance_scale, x_in
                                )[0]
                                / cutn_batches
                        )
                tv_losses = tv_loss(x_in)
                if use_secondary_model is True:
                    range_losses = range_loss(out)
                else:
                    range_losses = range_loss(out['pred_xstart'])
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = (
                        tv_losses.sum() * tv_scale
                        + range_losses.sum() * range_scale
                        + sat_losses.sum() * sat_scale
                )
                if init is not None and args.init_scale:
                    init_losses = lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * args.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if not torch.isnan(x_in_grad).any():
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    # print("NaN'd")
                    x_is_NaN = True
                    grad = torch.zeros_like(x)
            if args.clamp_grad and not x_is_NaN:
                magnitude = grad.square().mean().sqrt()
                return (
                        grad * magnitude.clamp(max=args.clamp_max) / magnitude
                )  # min=-0.02, min=-clamp_max,
            return grad

        if args.diffusion_sampling_mode == 'ddim':
            sample_fn = diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = diffusion.plms_sample_loop_progressive

        image_display = Output()
        for i in track(range(args.n_batches)):
            display.clear_output(wait=True)
            display.display(image_display)
            gc.collect()
            torch.cuda.empty_cache()
            cur_t = diffusion.num_timesteps - skip_steps - 1
            total_steps = cur_t

            if perlin_init:
                init = regen_perlin()

            if args.diffusion_sampling_mode == 'ddim':
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    eta=eta,
                )
            else:
                samples = sample_fn(
                    model,
                    (batch_size, 3, args.side_y, args.side_x),
                    clip_denoised=clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=True,
                    skip_timesteps=skip_steps,
                    init_image=init,
                    randomize_class=randomize_class,
                    order=2,
                )

            # with run_display:
            # display.clear_output(wait=True)
            for j, sample in enumerate(samples):
                cur_t -= 1
                intermediateStep = False
                if args.steps_per_checkpoint is not None:
                    if j % steps_per_checkpoint == 0 and j > 0:
                        intermediateStep = True
                elif j in args.intermediate_saves:
                    intermediateStep = True
                with image_display:
                    if (
                            j % args.display_rate == 0
                            or cur_t == -1
                            or intermediateStep == True
                    ):
                        for _, image in enumerate(sample['pred_xstart']):
                            percent = math.ceil(j / total_steps * 100)
                            if args.n_batches > 0:
                                # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                                if (
                                        cur_t == -1
                                        and args.intermediates_in_subfolder is True
                                ):
                                    filename = (
                                        f'{args.batch_name}({args.batchNum})_{i}.png'
                                    )
                                else:
                                    # If we're working with percentages, append it
                                    if args.steps_per_checkpoint is not None:
                                        filename = f'{args.batch_name}({args.batchNum})_{i:04}-{percent:02}%.png'
                                    # Or else, iIf we're working with specific steps, append those
                                    else:
                                        filename = f'{args.batch_name}({args.batchNum})_{i:04}-{j:03}.png'
                            image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                            if j % args.display_rate == 0 or cur_t == -1:
                                image.save('progress.png')
                                display.clear_output(wait=True)
                                display.display(display.Image('progress.png'))
                            if args.steps_per_checkpoint is not None:
                                if j % args.steps_per_checkpoint == 0 and j > 0:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            else:
                                if j in args.intermediate_saves:
                                    if args.intermediates_in_subfolder is True:
                                        image.save(f'{partialFolder}/{filename}')
                                    else:
                                        image.save(f'{batchFolder}/{filename}')
                            if cur_t == -1:
                                if frame_num == 0:
                                    save_settings()
                                image.save(f'{batchFolder}/{filename}')

            plt.plot(np.array(loss_values), 'r')


def save_settings():
    setting_list = {
        'text_prompts': text_prompts,
        'image_prompts': image_prompts,
        'clip_guidance_scale': clip_guidance_scale,
        'tv_scale': tv_scale,
        'range_scale': range_scale,
        'sat_scale': sat_scale,
        'cutn_batches': cutn_batches,
        'init_image': init_image,
        'init_scale': init_scale,
        'skip_steps': skip_steps,
        'frames_skip_steps': frames_skip_steps,
        'perlin_init': perlin_init,
        'perlin_mode': perlin_mode,
        'skip_augs': skip_augs,
        'randomize_class': randomize_class,
        'clip_denoised': clip_denoised,
        'clamp_grad': clamp_grad,
        'clamp_max': clamp_max,
        'seed': seed,
        'fuzzy_prompt': fuzzy_prompt,
        'rand_mag': rand_mag,
        'eta': eta,
        'width': width_height[0],
        'height': width_height[1],
        'diffusion_model': diffusion_model,
        'use_secondary_model': use_secondary_model,
        'steps': steps,
        'diffusion_steps': diffusion_steps,
        'diffusion_sampling_mode': diffusion_sampling_mode,
        'ViTB32': ViTB32,
        'ViTB16': ViTB16,
        'ViTL14': ViTL14,
        'RN101': RN101,
        'RN50': RN50,
        'RN50x4': RN50x4,
        'RN50x16': RN50x16,
        'RN50x64': RN50x64,
        'cut_overview': str(cut_overview),
        'cut_innercut': str(cut_innercut),
        'cut_ic_pow': cut_ic_pow,
        'cut_icgray_p': str(cut_icgray_p),
    }
    # print('Settings:', setting_list)
    with open(
            f"{batchFolder}/{batch_name}({batchNum})_settings.txt", "w+"
    ) as f:  # save settings
        json.dump(setting_list, f, ensure_ascii=False, indent=4)


# %%
# !! {"metadata": {
# !!    "cellView": "form",
# !!    "id": "DefSecModel"
# !! }}
# @title 1.6 Define the secondary diffusion model


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,
                            ConvBlock(cs[1], cs[2]),
                            ConvBlock(cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,
                                    ConvBlock(cs[2], cs[3]),
                                    ConvBlock(cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,
                                            ConvBlock(cs[3], cs[4]),
                                            ConvBlock(cs[4], cs[4]),
                                            SkipBlock(
                                                [
                                                    self.down,
                                                    ConvBlock(cs[4], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[4]),
                                                    self.up,
                                                ]
                                            ),
                                            ConvBlock(cs[4] * 2, cs[4]),
                                            ConvBlock(cs[4], cs[3]),
                                            self.up,
                                        ]
                                    ),
                                    ConvBlock(cs[3] * 2, cs[3]),
                                    ConvBlock(cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            ConvBlock(cs[2] * 2, cs[2]),
                            ConvBlock(cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]
            ),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


# %%
# !! {"metadata": {
# !!    "id": "DiffClipSetTop"
# !! }}
"""
# 2. Diffusion and CLIP model settings
"""

# %%
# !! {"metadata": {
# !!   "id": "ModelSettings"
# !!  }}
# @markdown ####**Models Settings:**
diffusion_model = "512x512_diffusion_uncond_finetune_008100"  # @param ["256x256_diffusion_uncond", "512x512_diffusion_uncond_finetune_008100"]
use_secondary_model = True  # @param {type: 'boolean'}
diffusion_sampling_mode = 'ddim'  # @param ['plms','ddim']

use_checkpoint = True  # @param {type: 'boolean'}
ViTB32 = True  # @param{type:"boolean"}
ViTB16 = True  # @param{type:"boolean"}
ViTL14 = False  # @param{type:"boolean"}
RN101 = False  # @param{type:"boolean"}
RN50 = True  # @param{type:"boolean"}
RN50x4 = False  # @param{type:"boolean"}
RN50x16 = False  # @param{type:"boolean"}
RN50x64 = False  # @param{type:"boolean"}

# @markdown If you're having issues with model downloads, check this to compare SHA's:
check_model_SHA = False  # @param{type:"boolean"}


def download_models(diffusion_model, use_secondary_model, fallback=False):
    model_256_downloaded = False
    model_512_downloaded = False
    model_secondary_downloaded = False

    model_256_SHA = '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_secondary_SHA = (
        '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a'
    )

    model_256_link = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt'
    model_512_link = 'https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link = (
        'https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth'
    )

    model_256_link_fb = (
        'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'
    )
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_link_fb = (
        'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth'
    )

    model_256_path = f'{model_path}/256x256_diffusion_uncond.pt'
    model_512_path = f'{model_path}/512x512_diffusion_uncond_finetune_008100.pt'
    model_secondary_path = f'{model_path}/secondary_model_imagenet_2.pth'

    if fallback:
        model_256_link = model_256_link_fb
        model_512_link = model_512_link_fb
        model_secondary_link = model_secondary_link_fb
    # Download the diffusion model
    if diffusion_model == '256x256_diffusion_uncond':
        if os.path.exists(model_256_path) and check_model_SHA:
            print('Checking 256 Diffusion File')
            with open(model_256_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_256_SHA:
                print('256 Model SHA matches')
                model_256_downloaded = True
            else:
                print("256 Model SHA doesn't match, redownloading...")
                wget(model_256_link, model_path)
                if os.path.exists(model_256_path):
                    model_256_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model, use_secondary_model, True)
        elif (
                os.path.exists(model_256_path)
                and not check_model_SHA
                or model_256_downloaded == True
        ):
            print(
                '256 Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            wget(model_256_link, model_path)
            if os.path.exists(model_256_path):
                model_256_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model, True)
    elif diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            print('Checking 512 Diffusion File')
            with open(model_512_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_512_SHA:
                print('512 Model SHA matches')
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model, use_secondary_model, True)
            else:
                print("512 Model SHA doesn't match, redownloading...")
                wget(model_512_link, model_path)
                if os.path.exists(model_512_path):
                    model_512_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model, use_secondary_model, True)
        elif (
                os.path.exists(model_512_path)
                and not check_model_SHA
                or model_512_downloaded == True
        ):
            print(
                '512 Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            wget(model_512_link, model_path)
            model_512_downloaded = True
    # Download the secondary diffusion model v2
    if use_secondary_model:
        if os.path.exists(model_secondary_path) and check_model_SHA:
            print('Checking Secondary Diffusion File')
            with open(model_secondary_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_secondary_SHA:
                print('Secondary Model SHA matches')
                model_secondary_downloaded = True
            else:
                print("Secondary Model SHA doesn't match, redownloading...")
                wget(model_secondary_link, model_path)
                if os.path.exists(model_secondary_path):
                    model_secondary_downloaded = True
                else:
                    print('First URL Failed using FallBack')
                    download_models(diffusion_model, use_secondary_model, True)
        elif (
                os.path.exists(model_secondary_path)
                and not check_model_SHA
                or model_secondary_downloaded == True
        ):
            print(
                'Secondary Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            wget(model_secondary_link, model_path)
            if os.path.exists(model_secondary_path):
                model_secondary_downloaded = True
            else:
                print('First URL Failed using FallBack')
                download_models(diffusion_model, use_secondary_model, True)


download_models(diffusion_model, use_secondary_model)

model_config = model_and_diffusion_defaults()
if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
    model_config.update(
        {
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': not useCPU,
            'use_scale_shift_norm': True,
        }
    )
elif diffusion_model == '256x256_diffusion_uncond':
    model_config.update(
        {
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps': 1000,  # No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing': 250,  # No need to edit this, it is taken care of later.
            'image_size': 256,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_checkpoint': use_checkpoint,
            'use_fp16': not useCPU,
            'use_scale_shift_norm': True,
        }
    )

if use_secondary_model:
    secondary_model = SecondaryDiffusionImageNet2()
    secondary_model.load_state_dict(
        torch.load(f'{model_path}/secondary_model_imagenet_2.pth', map_location='cpu')
    )
    secondary_model.eval().requires_grad_(False).to(device)

clip_models = []
if ViTB32 is True:
    clip_models.append(
        clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if ViTB16 is True:
    clip_models.append(
        clip.load('ViT-B/16', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if ViTL14 is True:
    clip_models.append(
        clip.load('ViT-L/14', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50 is True:
    clip_models.append(
        clip.load('RN50', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x4 is True:
    clip_models.append(
        clip.load('RN50x4', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x16 is True:
    clip_models.append(
        clip.load('RN50x16', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN50x64 is True:
    clip_models.append(
        clip.load('RN50x64', jit=False)[0].eval().requires_grad_(False).to(device)
    )
if RN101 is True:
    clip_models.append(
        clip.load('RN101', jit=False)[0].eval().requires_grad_(False).to(device)
    )

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)
lpips_model = lpips.LPIPS(net='vgg').to(device)

# %%
# !! {"metadata": {
# !!    "id": "SettingsTop"
# !! }}
"""
# 3. Settings
"""

# %%
# !! {"metadata": {
# !!    "id": "BasicSettings"
# !!  }}
# @markdown ####**Basic Settings:**
batch_name = 'TimeToDisco'  # @param{type: 'string'}
steps = 250  # @param [25,50,100,150,250,500,1000]{type: 'raw', allow-input: true}
width_height = [1280, 768]  # @param{type: 'raw'}
clip_guidance_scale = 5000  # @param{type: 'number'}
tv_scale = 0  # @param{type: 'number'}
range_scale = 150  # @param{type: 'number'}
sat_scale = 0  # @param{type: 'number'}
cutn_batches = 4  # @param{type: 'number'}
skip_augs = False  # @param{type: 'boolean'}

# @markdown ---

# @markdown ####**Init Settings:**
init_image = None  # @param{type: 'string'}  # can be a path or web url
init_scale = 1000  # @param{type: 'integer'}
skip_steps = 10  # @param{type: 'integer'}
# @markdown *Make sure you set skip_steps to ~50% of your steps if you want to use an init image.*

# Get corrected sizes
side_x = (width_height[0] // 64) * 64
side_y = (width_height[1] // 64) * 64
if side_x != width_height[0] or side_y != width_height[1]:
    print(
        f'Changing output size to {side_x}x{side_y}. Dimensions must by multiples of 64.'
    )

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
model_config.update(
    {
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    }
)

# Make folder for batch
batchFolder = f'{outDirPath}/{batch_name}'
createPath(batchFolder)

# @markdown ---


# @markdown `frame_skip_steps` will blur the previous frame - higher values will flicker less but struggle to add enough new detail to zoom into.
frames_skip_steps = '60%'  # @param ['40%', '50%', '60%', '70%', '80%'] {type: 'string'}


def split_prompts(prompts):
    prompt_series = pd.Series([np.nan])
    for i, prompt in prompts.items():
        prompt_series[i] = prompt
    # prompt_series = prompt_series.astype(str)
    prompt_series = prompt_series.ffill().bfill()
    return prompt_series


# %%
# !! {"metadata": {
# !!    "id": "ExtraSetTop"
# !! }}
"""
### Extra Settings
 Partial Saves, Advanced Settings, Cutn Scheduling
"""

# %%
# !! {"metadata": {
# !!   "id": "ExtraSettings"
# !! }}
# @markdown ####**Saving:**

intermediate_saves = 0  # @param{type: 'raw'}
intermediates_in_subfolder = True  # @param{type: 'boolean'}
# @markdown Intermediate steps will save a copy at your specified intervals. You can either format it as a single integer or a list of specific steps

# @markdown A value of `2` will save a copy at 33% and 66%. 0 will save none.

# @markdown A value of `[5, 9, 34, 45]` will save at steps 5, 9, 34, and 45. (Make sure to include the brackets)


if type(intermediate_saves) is not list:
    if intermediate_saves:
        steps_per_checkpoint = math.floor(
            (steps - skip_steps - 1) // (intermediate_saves + 1)
        )
        steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1
        print(f'Will save every {steps_per_checkpoint} steps')
    else:
        steps_per_checkpoint = steps + 10
else:
    steps_per_checkpoint = None

if intermediate_saves and intermediates_in_subfolder is True:
    partialFolder = f'{batchFolder}/partials'
    createPath(partialFolder)

    # @markdown ---

# @markdown ####**Advanced Settings:**
# @markdown *There are a few extra advanced settings available if you double click this cell.*

# @markdown *Perlin init will replace your init, so uncheck if using one.*

perlin_init = False  # @param{type: 'boolean'}
perlin_mode = 'mixed'  # @param ['mixed', 'color', 'gray']
set_seed = 'random_seed'  # @param{type: 'string'}
eta = 0.8  # @param{type: 'number'}
clamp_grad = True  # @param{type: 'boolean'}
clamp_max = 0.05  # @param{type: 'number'}

### EXTRA ADVANCED SETTINGS:
randomize_class = True
clip_denoised = False
fuzzy_prompt = False
rand_mag = 0.05

# @markdown ---

# @markdown ####**Cutn Scheduling:**
# @markdown Format: `[40]*400+[20]*600` = 40 cuts for the first 400 /1000 steps, then 20 for the last 600/1000

# @markdown cut_overview and cut_innercut are cumulative for total cutn on any given step. Overview cuts see the entire image and are good for early structure, innercuts are your standard cutn.

cut_overview = "[12]*400+[4]*600"  # @param {type: 'string'}
cut_innercut = "[4]*400+[12]*600"  # @param {type: 'string'}
cut_ic_pow = 1  # @param {type: 'number'}
cut_icgray_p = "[0.2]*400+[0]*600"  # @param {type: 'string'}

# %%
# !! {"metadata": {
# !!    "id": "Prompts"
# !! }}
text_prompts = {
    0: [
        "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
        "yellow color scheme",
    ],
}

image_prompts = {
    # 0:['ImagePromptsWorkButArentVeryGood.png:2',],
}

# %%
# !! {"metadata": {
# !!    "id": "DiffuseTop"
# !! }}
"""
# 4. Diffuse!
"""

# %%
# !! {"metadata": {
# !!    "id": "DoTheRun"
# !!  }}
# @title Do the Run!
# @markdown `n_batches` ignored with animation modes.
display_rate = 50  # @param{type: 'number'}
n_batches = 50  # @param{type: 'number'}

# Update Model Settings
timestep_respacing = f'ddim{steps}'
diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
model_config.update(
    {
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    }
)

batch_size = 1

skip_step_ratio = int(frames_skip_steps.rstrip("%")) / 100
calc_frames_skip_steps = math.floor(steps * skip_step_ratio)

if steps <= calc_frames_skip_steps:
    sys.exit("ERROR: You can't skip more steps than your total steps")

start_frame = 0
batchNum = len(glob(batchFolder + "/*.txt"))
while (
        os.path.isfile(f"{batchFolder}/{batch_name}({batchNum})_settings.txt") is True
        or os.path.isfile(f"{batchFolder}/{batch_name}-{batchNum}_settings.txt") is True
):
    batchNum += 1

print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')

if set_seed == 'random_seed':
    random.seed()
    seed = random.randint(0, 2 ** 32)
    # print(f'Using seed: {seed}')
else:
    seed = int(set_seed)

args = {
    'batchNum': batchNum,
    'prompts_series': split_prompts(text_prompts) if text_prompts else None,
    'image_prompts_series': split_prompts(image_prompts) if image_prompts else None,
    'seed': seed,
    'display_rate': display_rate,
    'n_batches': n_batches,
    'batch_size': batch_size,
    'batch_name': batch_name,
    'steps': steps,
    'diffusion_sampling_mode': diffusion_sampling_mode,
    'width_height': width_height,
    'clip_guidance_scale': clip_guidance_scale,
    'tv_scale': tv_scale,
    'range_scale': range_scale,
    'sat_scale': sat_scale,
    'cutn_batches': cutn_batches,
    'init_image': init_image,
    'init_scale': init_scale,
    'skip_steps': skip_steps,
    'side_x': side_x,
    'side_y': side_y,
    'timestep_respacing': timestep_respacing,
    'diffusion_steps': diffusion_steps,
    'max_frames': 1,
    'start_frame': start_frame,
    'skip_step_ratio': skip_step_ratio,
    'calc_frames_skip_steps': calc_frames_skip_steps,
    'text_prompts': text_prompts,
    'image_prompts': image_prompts,
    'cut_overview': eval(cut_overview),
    'cut_innercut': eval(cut_innercut),
    'cut_ic_pow': cut_ic_pow,
    'cut_icgray_p': eval(cut_icgray_p),
    'intermediate_saves': intermediate_saves,
    'intermediates_in_subfolder': intermediates_in_subfolder,
    'steps_per_checkpoint': steps_per_checkpoint,
    'perlin_init': perlin_init,
    'perlin_mode': perlin_mode,
    'set_seed': set_seed,
    'eta': eta,
    'clamp_grad': clamp_grad,
    'clamp_max': clamp_max,
    'skip_augs': skip_augs,
    'randomize_class': randomize_class,
    'clip_denoised': clip_denoised,
    'fuzzy_prompt': fuzzy_prompt,
    'rand_mag': rand_mag,
}

args = SimpleNamespace(**args)

print('Prepping model...')
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(
    torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu')
)
model.requires_grad_(False).eval().to(device)
for name, param in model.named_parameters():
    if 'qkv' in name or 'norm' in name or 'proj' in name:
        param.requires_grad_()
if model_config['use_fp16']:
    model.convert_to_fp16()

gc.collect()
torch.cuda.empty_cache()
try:
    do_run()
except KeyboardInterrupt:
    pass
finally:
    print('Seed used:', seed)
    gc.collect()
    torch.cuda.empty_cache()
