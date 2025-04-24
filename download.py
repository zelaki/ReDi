
"""
Functions for downloading pre-trained SiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'SiT-XL-2-256x256.pt'}


# def find_model(model_name):
#     """
#     Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
#     """

#     assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
#     checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
#     if "ema" in checkpoint:  # supports checkpoints from train.py
#         checkpoint = checkpoint["ema"]
#     return checkpoint


def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  
        return download_model(model_name)
    else:  
        assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dropbox.com/scl/fi/yi726j26yc57s4qhzgbtt/3000000.pt?rlkey=tcr8e0n9rrm12wfen44dkz00r&e=1&st=59cyam58&dl=1'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model
