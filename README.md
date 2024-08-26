# Latent-SDE
This is the official implementation for [Latent-SDE: guiding stochastic differential equations in latent space for unpaired image-to-image translation](https://link.springer.com/article/10.1007/s40747-024-01566-1).
# Datasets
Please download the AFHQ and CelebA-HQ dataset following the dataset instructions in [https://github.com/clovaai/stargan-v2](https://github.com/clovaai/stargan-v2) and put them in `data/`. We also provide some demo images in `data/` for quick start.
# Pretrained Models
All used pretrained models can be downloaded from here. Please put them in `pretrained_model/`. The `ddpm_dog_64x64.pt` and `ddpm_female_64x64.pt` are the pretrained diffusion models based on [ddpm]( https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-) on dog on AFHQ and female on CelebA-HQ respectively. `dse_cat2dog_64x64.pt`, `dse_wild2dog_64x64.pt` and `dse_male2female_64x64.pt` are pretrained classifier for domain-specific extractor on cat2dog, wild2dog and male2female task respectively. `die_cat_64x64.pt`, `die_wild_64x64.pt` and `die_male_64x64.pt` are pretrained domain-independent extractor on cat and wild on AFHQ and male on CelebA-HQ respectively. `64x64_classifier.pt` is the pretrained classifier on ImageNet provided in [guided-diffusion](https://github.com/openai/guided-diffusion) used for initial weight of classifier.
