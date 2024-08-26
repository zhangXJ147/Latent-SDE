import os
import logging
import importlib
import numpy as np
import torch
import torch.utils.data as data
from model.ddpm import Model
from datasets import get_dataset, rescale, inverse_rescale
import torchvision.utils as tvu
from functions.denoising import latent_sde_sample
from model.VQGAN.vqgan import VQModel
from model.U_net import UNet
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from functions.resizer import Resizer
from guided_diffusion.script_util import create_model, create_dse


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def cleanup_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if 'module' in k:
            new_name = k[7:]
        else:
            new_name = k
        new_state[new_name] = v
    return new_state


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def truncation(global_step, ratio=0.5):
    part = int(global_step * ratio)
    weight_l = torch.zeros(part).reshape(-1, 1)
    weight_r = torch.ones(global_step - part).reshape(-1, 1)
    weight = torch.cat((weight_l, weight_r), dim=0)
    return weight


class Latent_SDE(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        # load vqgan
        self.vqgan = VQModel(**vars(self.config.model.VQGAN.params)).eval().to(self.device)
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {self.config.model.VQGAN.params.ckpt_path}")

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    @torch.no_grad()
    def encode(self, x):
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.config.model.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent):
        model = self.vqgan
        if self.config.model.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)
        out = model.decode(x_latent_quant)
        return out

    def latent_sde(self, ):
        args, config = self.args, self.config
        # load SDE
        if args.diffusionmodel == 'ADM':
            model = create_model(image_size=config.data.image_size,
                                 num_class=config.model.num_class,
                                 num_channels=config.model.num_channels,
                                 num_res_blocks=config.model.num_res_blocks,
                                 learn_sigma=config.model.learn_sigma,
                                 class_cond=config.model.class_cond,
                                 attention_resolutions=config.model.attention_resolutions,
                                 num_heads=config.model.num_heads,
                                 num_head_channels=config.model.num_head_channels,
                                 num_heads_upsample=config.model.num_heads_upsample,
                                 use_scale_shift_norm=config.model.use_scale_shift_norm,
                                 dropout=config.model.dropout,
                                 resblock_updown=config.model.resblock_updown,
                                 use_fp16=config.model.use_fp16,
                                 use_new_attention_order=config.model.use_new_attention_order)
            states = torch.load(args.ckpt, map_location='cpu')
            model.load_state_dict(states)
            model = model.to(self.device)
            # model = torch.nn.DataParallel(model)
            model.eval()
        elif args.diffusionmodel == 'DDPM':
            model = Model(config)
            states = torch.load(self.args.ckpt, map_location='cpu')
            states = cleanup_state_dict(states)
            model = model.to(self.device)
            # model = torch.nn.DataParallel(model)
            model.load_state_dict(states, strict=True)
            model.eval()
        else:
            raise ValueError(f"unsupported diffusion model")

        # load domain-specific feature extractor
        logger.log("loading dse...")
        dse = create_dse(image_size=config.model.image_size,
                         num_class=config.dse.num_class,
                         classifier_use_fp16=config.dse.classifier_use_fp16,
                         classifier_width=config.dse.classifier_width,
                         classifier_depth=config.dse.classifier_depth,
                         classifier_attention_resolutions=config.dse.classifier_attention_resolutions,
                         classifier_use_scale_shift_norm=config.dse.classifier_use_scale_shift_norm,
                         classifier_resblock_updown=config.dse.classifier_resblock_updown,
                         classifier_pool=config.dse.classifier_pool,
                         phase=args.phase)
        dse.load_state_dict(
            dist_util.load_state_dict(config.dse.classifier_path, map_location="cpu")
        )
        dse.to(self.device)
        if config.dse.classifier_use_fp16:
            dse.convert_to_fp16()
        dse.eval()

        # load domain-independent feature extractor
        logger.log("loading die...")
        die = UNet(in_channels=config.data.channels, out_channels=config.data.channels)
        states = torch.load(args.diepath, map_location='cpu')
        die.load_state_dict(states)
        cont = die.to(self.device)
        cont.eval()

        # create dataset
        dataset = get_dataset(phase=args.phase, image_size=config.data.image_size, data_path=args.testdata_path)
        data_loader = data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

        for it in range(args.sample_step):
            for i, (y, label, name) in enumerate(data_loader):
                logging.info(f'batch:{i}/{len(dataset) / args.batch_size}')
                n = y.size(0)
                y0 = rescale(y).to(self.device)
                y0 = self.encode(y0)
                # let x0 be source image
                x0 = y0
                total_noise_levels = args.t
                a = (1 - self.betas).cumprod(dim=0)
                e = torch.randn_like(y0)
                y = y0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
                for i in reversed(range(total_noise_levels)):
                    print(i)
                    y_ = y
                    for _ in range(args.travel_step):
                        e = torch.randn_like(y0)
                        # the start point M: y ∼ qM|0(y|x0)
                        t = (torch.ones(n) * i).to(self.device)
                        # sample perturbed source image from the perturbation kernel: x ∼ qs|0(x|x0)
                        xt = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        # latent-sde update
                        y_ = latent_sde_sample(y=y_, xt=xt, t=t, dse=dse, model=model, die=die, ls=args.ls,
                                               li=args.li, logvar=self.logvar, betas=self.betas, s1=args.s1,
                                               s2=args.s2, model_name=args.diffusionmodel)
                        y = y_
                        # time travel
                        e = torch.randn_like(y0)
                        y_ = y_ * extract((1 - self.betas).sqrt(), t, y_.shape) + \
                             extract(self.betas.sqrt(), t, y_.shape) * e

                y = self.decode(y)
                y = inverse_rescale(y)
                # save image
                for b in range(n):
                    os.makedirs(os.path.join(self.args.samplepath, str(it)), exist_ok=True)
                    tvu.save_image(
                        y[b], os.path.join(self.args.samplepath, str(it), name[b])
                    )

        logging.info('Finshed sampling.')
