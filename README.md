# Latent-SDE
This is the official implementation for [Latent-SDE: guiding stochastic differential equations in latent space for unpaired image-to-image translation](https://link.springer.com/article/10.1007/s40747-024-01566-1).
## Datasets
Please download the AFHQ and CelebA-HQ dataset following the dataset instructions in [https://github.com/clovaai/stargan-v2](https://github.com/clovaai/stargan-v2) and put them in `data/`. We also provide some demo images in `data/` for quick start.
## Pretrained Models
All used pretrained models can be downloaded from [here](https://pan.baidu.com/s/1eJ_EQV4wJqMjYVzKsonTMg?pwd=4jjd). Please put them in `pretrained_model/`. The `VQGAN/model_f4.ckpt` is provided by [latent-diffusion]( https://ommer-lab.com/files/latent-diffusion/vq-f4.zip). `ddpm_dog_64x64.pt` and `ddpm_female_64x64.pt` are the pretrained diffusion models on dog on AFHQ and female on CelebA-HQ respectively. `dse_cat2dog_64x64.pt`, `dse_wild2dog_64x64.pt` and `dse_male2female_64x64.pt` are pretrained classifier for domain-specific extractor on cat2dog, wild2dog and male2female task respectively. `die_cat_64x64.pt`, `die_wild_64x64.pt` and `die_male_64x64.pt` are pretrained domain-independent extractor on cat and wild on AFHQ and male on CelebA-HQ respectively. `64x64_classifier.pt` is the pretrained classifier on ImageNet provided in [guided-diffusion](https://github.com/openai/guided-diffusion) used for initial weight of classifier.
## Run Latent-SDE for Two-Domain Image Translation
    $ python run_Latent_SDE.py
## Evaluation
    $ python run_score.py
## Training Domain-specific Extractors
    $ python run_train_dse.py
## Training Domain-independent Extractors
    $ python run_train_die.py
## Re-training Score-based Diffusion Model
Firstly, please replace Diffusion/Model.py in [Ddpm]( https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-) with model/ddpm.py in Latent-SDE. Secondly, please create code that combines the encoder E of VQGAN with GaussianDiffusionTrainer in Ddpm/Diffusion/Difussion.py. Finally, please use Main.py in Ddpm for training. Thanks.
## References
If you find this repository helpful, please cite as:
```
@article{zhang2024latent,
    title={Latent-SDE: guiding stochastic differential equations in latent space for unpaired image-to-image translation},
    author={Zhang, Xianjie and Li, Min and He, Yujie and Gou, Yao and Zhang, Yusen},
    journal={Complex \& Intelligent Systems},
    pages={1--11},
    year={2024},
    publisher={Springer}
}
```
This implementation is based on [EGSDE](https://github.com/ML-GSAI/EGSDE/tree/master) and [latent-diffusion](https://github.com/CompVis/latent-diffusion).
