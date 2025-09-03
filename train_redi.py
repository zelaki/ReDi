# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from torchvision.transforms import Normalize

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from dataset import CustomDataset

from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 



CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)




def preprocess_raw_image(x):
    resolution = x.shape[-1]
    x = x / 255.
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')

    return x


@torch.no_grad()
def load_encoders(device, resolution=256):
    assert (resolution == 256) or (resolution == 512)
    encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14_reg')
    del encoder.head
    encoder.head = torch.nn.Identity()
    encoder = encoder.to(device)
    encoder.eval()
    
    return encoder


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        experiment_name = f"{experiment_index:03d}-{model_string_name}-" \
                        f"{args.path_type}-{args.prediction}"
        experiment_dir = f"{args.results_dir}/{experiment_name}"  # Create an experiment folder
        experiment_dir += f"dino-pca-{args.pca_rank}"
        if args.loss_weight:
            experiment_dir += f"loss-weight-{args.loss_weight}"
        if args.dino_drop_prob:
            experiment_dir += f"dino-drop-prob-{args.dino_drop_prob}"

        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=4+args.pca_rank,
        repa_loss=args.repa_loss,
        repa_layer=args.repa_layer,
        learn_sigma=False

    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

    requires_grad(ema, False)
    
    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = CustomDataset(args.feature_path)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    encoder = load_encoders(device=device)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)


    pca_stats = torch.load(args.pca_model_path)
    pca_model = pca_stats["pca_model"]

    pca_components = torch.tensor(pca_model.components_[:args.pca_rank, :]).to(device)

    pca_mean = torch.tensor(pca_model.mean_).to(device)
    mean_dino = torch.tensor(pca_stats["mean"]).to(device)
    std_dino = torch.tensor(pca_stats["std"]).to(device)



    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for raw_image, x, y in loader:


            raw_image = raw_image.to(device)

            with torch.no_grad():

                raw_image_ = preprocess_raw_image(raw_image)

                z = encoder.forward_features(raw_image_)['x_norm_patchtokens']
                if args.repa_loss:
                    dino_feats = z.clone()
                else:
                    dino_feats = None

                z = (z - mean_dino) / std_dino
                z = z - pca_mean

                z_pca = z @ pca_components.T
                z_pca = z_pca.reshape(x.shape[0], 16, 16, args.pca_rank).permute(0,3,1,2)
                z_pca = torch.nn.functional.interpolate(z_pca, size=(32,32), mode='bilinear')


            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)

            x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)

            x_joint = torch.cat([x,z_pca], dim=1)


            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(
                model, x_joint, model_kwargs,
                dino_drop_prob=args.dino_drop_prob if args.dino_drop_prob is not None else 0,
                dino_feats=dino_feats)

            if args.repa_loss:
                loss_denoise = loss_dict["loss"].mean()
                loss_proj = loss_dict["proj_loss"].mean()
                loss = loss_denoise + loss_proj * args.repa_weight
            else:
                loss = loss_dict["loss"].mean()
            
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps > args.max_train_steps:
                break


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--pca-rank", type=int, default=8)

    parser.add_argument("--repa-loss", type=bool, default=False)
    parser.add_argument("--repa-weight", type=float, default=0.5)
    parser.add_argument("--repa-layer", type=float, default=8)


    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--sample-every", type=int, default=10_000)
    parser.add_argument("--dino-drop-prob", type=float, default=None)
    parser.add_argument("--pca-model-path", type=str, default="./pcs/dino_pca_model.pth")
    parser.add_argument("--max-train-steps", type=int, default=400_000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
