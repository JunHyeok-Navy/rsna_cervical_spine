## Basic Library
import gc
import os
import cv2
import config
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union

## Processing Dicom File
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from skimage import exposure
from skimage.transform import resize

## Pytorch
import torch
from torch import Tensor
import torch.nn.functional as F

## Helper
from tqdm import tqdm
from joblib import Parallel, delayed

## Interpolate Volume Function
def interpolate_volume(volume: Tensor, vol_size: Optional[Tuple[int, int, int]] = None) -> Tensor:
    """Interpolate volume in last (Z) dimension
    >>> vol = torch.rand(64, 64, 12)
    >>> vol2 = interpolate_volume(vol)
    >>> vol2.shape
    torch.Size([64, 64, 64])
    >>> vol2 = interpolate_volume(vol, vol_size=(64, 64, 24))
    >>> vol2.shape
    torch.Size([64, 64, 24])
    """
    vol_shape = tuple(volume.shape)
    if not vol_size:
        d_new = min(vol_shape[:2])
        vol_size = (vol_shape[0], vol_shape[1], d_new)
    # assert vol_shape[0] == vol_shape[1], f"mixed shape: {vol_shape}"
    if vol_shape == vol_size:
        return volume
    return F.interpolate(volume.unsqueeze(0).unsqueeze(0), size=vol_size, mode="trilinear", align_corners=False)[0, 0]


## Read Dicom File to numpy data
def read_dicom(file_path):
    sample = pydicom.dcmread(file_path)
    
    ## Preprocessing
    image = apply_voi_lut(sample.pixel_array, sample)
    image = cv2.resize(image, config.VOLUMN_SIZE[:2], interpolation=cv2.INTER_LINEAR)
    return image

## Change image numpy data to volumn data
def imgs_to_volumn(file_path):
    dcm_files = glob(os.path.join(file_path, '*.dcm'))
    image_list = []
    for i in range(len(dcm_files)):
        try:
            sample_path = os.path.join(file_path, str(i+1) + '.dcm')
            sample_image = read_dicom(sample_path)
            image_list.append(sample_image)
        except:
            pass
    volumn = torch.tensor(image_list, dtype=torch.float32)
    ## Change Shape (D x W x H -> W x H x D)
    volumn = (volumn - volumn.min()) / float(volumn.max() - volumn.min())
    volumn = interpolate_volume(volumn, config.VOLUMN_SIZE).numpy()
    volumn = exposure.equalize_adapthist(volumn, kernel_size=np.array([64, 64, 64]), clip_limit=0.01)
    volumn = np.clip(volumn * 255, 0, 255).astype(np.uint8)
    return volumn

## Multiprocessing
def save_numpy(sample_path):
    sample_volumn = imgs_to_volumn(sample_path + '\\')

    ## Save Data to .npy file
    patient_id = sample_path.split('\\')[-1]
    sample_path = os.path.join(config.SAVE_PATH, patient_id)
    np.savez_compressed(sample_path, sample_volumn)

    ## Memory Usage
    del sample_volumn
    gc.collect()
    return None

Parallel(n_jobs=config.CORES)(delayed(save_numpy)(patient) for patient in tqdm(config.PATIENTS_LISTS, leave=True))