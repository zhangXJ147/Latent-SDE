import os
import yaml
from tool.utils import available_devices,format_devices
import argparse
import importlib
import torch as th
import torch.nn.functional as F
from torch.optim import AdamW
from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from datasets import get_dataset
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
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = UNet(in_channels=config.data.channels, out_channels=config.data.channels)
    model.to(device)

    resume_step = 0

    if args.pretrained:
        model = load_share_weights(model, args.pretrained_model)

    if args.resmue:
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_model, map_location=device
            )
        )

    mp_trainer = MixedPrecisionTrainer(
        model=model, initial_lg_loss_scale=16.0
    )

    # load low-pass filter
    shape = (args.batch_size, config.data.channels, config.data.image_size, config.data.image_size)
    shape_d = (
        args.batch_size, config.data.channels, int(config.data.image_size / args.down_N), int(config.data.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(device)
    up = Resizer(shape_d, args.down_N).to(device)

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
    val_data = None
    logger.log(f"creating optimizer...")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resmue:
        logger.log(f"loading optimizer state from checkpoint: {args.resume_opt}")
        states = dist_util.load_state_dict(args.resume_opt, map_location=device)
        opt.load_state_dict(states)
        resume_step = states['state'][0]['step'] - 1
        logger.log("start_step:{}".format(resume_step))
    logger.log("training die model...")

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
        data, _, _ = next(iter(data_loader))
        data = 2 * data - 1.0
        data = data.to(device)
        x = encode(data)
        x_ = up(down(data))
        x = model(x.detach())
        x = decode(x)
        loss = F.mse_loss(x, x_.detach())

        losses = {}
        losses[f"{phase}_loss"] = loss.detach()
        log_loss_dict(losses)
        del losses
        loss = loss.mean()
        mp_trainer.zero_grad()
        mp_trainer.backward(loss)

    for step in range(args.iterations - resume_step):
        logger.logkv("step", (step + 1) + resume_step)
        logger.logkv(
            "samples", (step + 1)
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data, phase=args.phase)
        mp_trainer.optimize(opt)
        if val_data is not None and not (step + 1) % args.eval_interval:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, phase="val")
                    model.train()
        if not (step + 1) % args.log_interval:
            logger.dumpkvs()
        if (step + 1) % args.save_interval == 0 and (step + 1) >= int(args.iterations / 4):
            logger.log("saving model...")
            save_model(model, opt, (step + 1) + resume_step)
    logger.log('training complete ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))


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
        batch_size=8,
        microbatch=-1,
        down_N=8,
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
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/train/cat']
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/train/wild']
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/train/male']
        args.num_class = 2
        args.iterations = 5000
        with open('', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)

