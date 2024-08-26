import os
import yaml
from tool.utils import available_devices,format_devices
import argparse
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
    classifier_and_diffusion_defaults,
    create_DSE_and_diffusion
)
from model.VQGAN.vqgan import VQModel
import torch
from tool.utils import dict2namespace

def load_share_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

from guided_diffusion.train_util import log_loss_dict
import datetime
def main(args, config):
    device = torch.device('cpu') if torch.cuda.is_available() else torch.device('cpu')
    model, diffusion = create_DSE_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    model.to(device)

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )
    resume_step = 0

    if args.pretrained:
        model = load_share_weights(model,args.pretrained_model)

    if args.resmue:
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=device
            )
        )

    # mp_trainer = MixedPrecisionTrainer(
    #     model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    # )

    # model = torch.nn.DataParallel(model)
    # load vqgan
    vqgan = VQModel(**vars(config.model.VQGAN.params)).eval().to(device)
    for param in vqgan.parameters():
        param.requires_grad = False
    print(f"load vqgan from {config.model.VQGAN.params.ckpt_path}")

    dataset = get_dataset(phase=args.phase, image_size=config.data.image_size, data_path=args.data_path)
    import torch.utils.data as data
    data = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    # val_data = None
    # logger.log(f"creating optimizer...")
    # opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    # if args.resmue:
    #     logger.log(f"loading optimizer state from checkpoint: {args.resume_opt}")
    #     states = dist_util.load_state_dict(args.resume_opt, map_location=device)
    #     opt.load_state_dict(states)
    #     resume_step = states['state'][0]['step'] - 1
    #     logger.log("start_step:{}".format(resume_step))
    # logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, labels, _ = data_loader
        labels = labels.long()
        labels = labels.to(device)
        batch = 2 * batch - 1.0
        batch = batch.to(device)
        with torch.no_grad():
            batch = encode(batch)

        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], device)
            batch = diffusion.q_sample(batch.detach(), t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=device)

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            logits = model(sub_batch, timesteps=sub_t)
            # loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {f"{prefix}_acc@1": compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )}
            # losses[f"{prefix}_loss"] = loss.detach()
            # log_loss_dict(diffusion, sub_t, losses)
            # del losses
            correct_val = losses[f"{prefix}_acc@1"].sum().numpy()
            logger.log('correct_val: ', correct_val)
            return correct_val
            # loss = loss.mean()
            # if loss.requires_grad:
            #     if i == 0:
            #         mp_trainer.zero_grad()
            #     mp_trainer.backward(loss * len(sub_batch) / len(batch))

    @torch.no_grad()
    def encode(x):
        model = vqgan
        x_latent = model.encoder(x)
        if not config.model.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        return x_latent

    correct_val = 0
    total_val = 0

    for step, val_data in enumerate(data):
        # logger.logkv("step", (step + 1) + resume_step)
        # logger.logkv(
        #     "samples", (step + 1)
        # )
        # if args.anneal_lr:
        #     set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        # forward_backward_log(data)
        # mp_trainer.optimize(opt)
        if val_data is not None:
            with th.no_grad():
                # with model.no_sync():
                    model.eval()
                    correct_val += forward_backward_log(val_data, prefix="val")
            total_val += len(val_data[0])
        print('correct_val: ', correct_val)
        print('total_val: ', total_val)
                    # model.train()
        # if not (step + 1) % args.log_interval:
        #     logger.dumpkvs()
        # if (step + 1) % args.save_interval == 0 and (step + 1) >= int(args.iterations / 4):
        #     logger.log("saving model...")
        #     save_model(mp_trainer, opt, (step + 1) + resume_step)
    acc_avg = correct_val / total_val
    logger.log('acc_avg: ', acc_avg)

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    th.save(
        mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
        os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
    )
    th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))

    # th.save(
    #     mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
    #     os.path.join(logger.get_dir(), f"model.pt"),
    # )
    # th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        dataset='cat2dog', # wild2dog/cat2dog/male2female/afhq
        data_path=['data/afhq/train/cat', 'data/afhq/train/dog'],
        pretrained=False,
        pretrained_model='pretrained_model/256x256_classifier.pt',
        resmue=True,
        val_data_dir="",
        noised=True,
        iterations=5000,
        lr=3e-4,
        weight_decay=0.05,
        anneal_lr=True,
        batch_size=32,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=50,
        eval_interval=50,
        save_interval=50,
        phase='train'
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    dataset = 'male2female' #cat2dog/wild2dog/male2female/multi_afhq(mutli-domain)
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'classifier_acc@1')
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/val/cat', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_cat2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_wild2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male', 'data/celeba_hq/val/female']
        args.resume_checkpoint = 'pretrained_model/dse_male2female_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'multi_afhq':
        args.data_path = ['data/afhq/train/cat','data/afhq/train/wild', 'data/afhq/train/dog']
        args.num_class = 3
        args.iterations = 10000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)

    dataset = 'cat2dog' #cat2dog/wild2dog/male2female/multi_afhq(mutli-domain)
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'classifier_acc@1')
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/val/cat', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_cat2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_wild2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male', 'data/celeba_hq/val/female']
        args.resume_checkpoint = 'pretrained_model/dse_male2female_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'multi_afhq':
        args.data_path = ['data/afhq/train/cat','data/afhq/train/wild', 'data/afhq/train/dog']
        args.num_class = 3
        args.iterations = 10000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)

    dataset = 'cat2dog' #cat2dog/wild2dog/male2female/multi_afhq(mutli-domain)
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'classifier_acc@1')
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/val/cat', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_cat2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_wild2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male', 'data/celeba_hq/val/female']
        args.resume_checkpoint = 'pretrained_model/dse_male2female_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'multi_afhq':
        args.data_path = ['data/afhq/train/cat','data/afhq/train/wild', 'data/afhq/train/dog']
        args.num_class = 3
        args.iterations = 10000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)

    dataset = 'wild2dog' #cat2dog/wild2dog/male2female/multi_afhq(mutli-domain)
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'classifier_acc@1')
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/val/cat', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_cat2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_wild2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male', 'data/celeba_hq/val/female']
        args.resume_checkpoint = 'pretrained_model/dse_male2female_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'multi_afhq':
        args.data_path = ['data/afhq/train/cat','data/afhq/train/wild', 'data/afhq/train/dog']
        args.num_class = 3
        args.iterations = 10000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)

    dataset = 'wild2dog' #cat2dog/wild2dog/male2female/multi_afhq(mutli-domain)
    #defalut args
    args = create_argparser().parse_args()
    args.dataset = dataset
    dir = os.path.join('runs', args.dataset, 'classifier_acc@1')
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger.configure(dir=dir, log_suffix=now)
    if dataset == 'cat2dog':
        args.data_path = ['data/afhq/val/cat', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_cat2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'wild2dog':
        args.data_path = ['data/afhq/val/wild', 'data/afhq/val/dog']
        args.resume_checkpoint = 'pretrained_model/dse_wild2dog_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/wild2dog/wild2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'male2female':
        args.data_path = ['data/celeba_hq/val/male', 'data/celeba_hq/val/female']
        args.resume_checkpoint = 'pretrained_model/dse_male2female_64x64.pt'
        args.num_class = 2
        args.iterations = 5000
        with open('profiles/male2female/male2female.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    if dataset == 'multi_afhq':
        args.data_path = ['data/afhq/train/cat','data/afhq/train/wild', 'data/afhq/train/dog']
        args.num_class = 3
        args.iterations = 10000
        with open('profiles/cat2dog/cat2dog.yml', "r") as f:
            config_ = yaml.load(f, Loader=yaml.FullLoader)
        config = dict2namespace(config_)
    main(args, config)