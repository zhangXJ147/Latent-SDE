import os
import yaml
from tool.utils import available_devices,format_devices
import argparse
import importlib
import torch as th
import torch.nn.functional as F
from torch.optim import AdamW
import cv2 as cv
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from datasets import get_dataset, inverse_rescale
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    diffusion_defaults,
    create_gaussian_diffusion)
from model.VQGAN.vqgan import VQModel
from model.U_net import UNet
from functions.resizer import Resizer
import torch
from tool.utils import dict2namespace
import torchvision.utils as tvu
from omegaconf import OmegaConf
import datetime

def load_share_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


def main(args, config):
    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')

    resume_step = 0

    # load low-pass filter
    shape = (args.batch_size, config.data.channels, config.data.image_size, config.data.image_size)
    shape_d = (
        args.batch_size, config.data.channels, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
    down1 = Resizer(shape, 1 / args.down_N).to(device)
    up1 = Resizer(shape_d, args.down_N).to(device)

    # load ldm
    vqgan = VQModel(**vars(config.model.VQGAN.params)).eval().to(device)
    for param in vqgan.parameters():
        param.requires_grad = False
    print(f"load vqgan from {config.model.VQGAN.params.ckpt_path}")

    dataset = get_dataset(phase=args.phase, image_size=config.data.image_size, data_path=args.data_path)
    import torch.utils.data as data
    data = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    config.data.image_size = 64
    args.down_N = 8
    shape = (args.batch_size, config.data.channels, config.data.image_size, config.data.image_size)
    shape_d = (
        args.batch_size, config.data.channels, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
    down2 = Resizer(shape, 1 / args.down_N).to(device)
    up2 = Resizer(shape_d, args.down_N).to(device)

    @torch.no_grad()
    def encode(x):
        model = vqgan
        x_latent = model.encoder(x)
        if not config.model.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        return x_latent

    def decode(x_latent):
        model = vqgan
        if config.model.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    def forward_backward_log(data_loader, phase="train"):
        data, _, name = next(iter(data_loader))
        data = 2 * data - 1.0
        data = data.to(device)
        # x_ = up1(down1(data))
        x_ = torch.tensor(cv.Laplacian(data.squeeze().permute(2, 1, 0).numpy(), -1, ksize=3)).permute(2, 1, 0).unsqueeze(0).to(device)
        x = encode(data)
        # x = up2(down2(x))
        x = torch.tensor(cv.Laplacian(x.squeeze().permute(2, 1, 0).numpy(), -1, ksize=3)).permute(2, 1, 0).unsqueeze(0).to(device)
        x = decode(x)
        return x_, x, name

    for step in range(args.iterations - resume_step):
        x0, x1, name = forward_backward_log(data, phase=args.phase)
        x0 = inverse_rescale(x0)
        x1 = inverse_rescale(x1)
        for b in range(x0.size(0)):
            os.makedirs(os.path.join(args.samplepath, 'high'), exist_ok=True)
            tvu.save_image(
                x0[b], os.path.join(args.samplepath, 'high', name[b])
            )
        for b in range(x1.size(0)):
            os.makedirs(os.path.join(args.samplepath, 'high_ED'), exist_ok=True)
            tvu.save_image(
                x1[b], os.path.join(args.samplepath, 'high_ED', name[b])
            )

def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(model, opt, step):
    th.save(
        model.state_dict(),
        os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
    )
    th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

    # th.save(
    #     mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
    #     os.path.join(logger.get_dir(), f"model.pt"),
    # )
    # th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt.pt"))


def create_argparser():
    defaults = dict(
        dataset='cat2dog', # wild2dog/cat2dog/male2female
        data_path=['data/afhq/train/cat'],
        pretrained=False,
        pretrained_model='',
        resmue=False,
        val_data_dir="",
        iterations=5000,
        lr=3e-4,
        weight_decay=0.05,
        anneal_lr=True,
        batch_size=1,
        microbatch=-1,
        down_N=16,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=50,
        eval_interval=50,
        save_interval=50,
        phase='train'
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    dataset = 'cat2dog' #cat2dog/wild2dog/male2female
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'die')
    args.samplepath = dir
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/test']
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild']
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male']
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)
