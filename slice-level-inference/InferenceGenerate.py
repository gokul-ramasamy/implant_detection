# Importing the required packages
import pandas as pd
import torch
import os
import datetime
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from VAE.models import VQVAE_BN as VQVAE

from piqa import SSIM
from vit_pytorch import ViT, MAE_Recon
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import cv2
import copy

def get_data_inference(csv_path):
    df = pd.read_csv(csv_path)
    data_dict = dict()
    for i in df.index.values.tolist():
        path = df.loc[i, "Path"]
        ind = path.split("/")[-1].split('.')[0]
        data_dict[ind] = path
    return data_dict

def min_max_norm(img, per_channel=False):
    if per_channel == False:
        norm = copy.deepcopy(img)
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                for i in range(img.shape[2]):
                    norm[:,:,i] = (norm[:,:,i] - np.amin(norm[:,:,i]))/(np.amax(norm[:,:,i]) - np.amin(norm[:,:,i]))
            else:
                raise Exception("RGB Image for normalisation has more than three channels")
        elif len(img.shape) == 2:
            norm = (norm[:,:] - np.amin(norm[:,:]))/(np.amax(norm[:,:]) - np.amin(norm[:,:]))
        else:
            raise Exception("Image for normalisation not grayscale or rgb")
    
    else:
        norm = copy.deepcopy(img)
        norm = (norm - np.amin(norm))/(np.amax(norm) - np.amin(norm))
    
    return norm

# Function to remove the zero pad
def remove_zero_pad(image):
    dummy = np.argwhere(image != 0) # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image

def large_contour(image):
    # Threshold the image
    thresh = image>0.001
    thresh = thresh.astype(np.uint8)

    # Finding the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
    else:
        return image
    # Cropping the image accordingly
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image


class CTVITDataset(Dataset):
    def __init__(self, data_dict, transform, image_size):
        self.data_dict = data_dict
        self.transform = transform
        self.image_size = image_size


    # Overriding the __len__ method as the number of images in the root folder
    def __len__(self):
        return len(self.data_dict)
    
    # Overriding the __getitem__ method to get the images in the root folder
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Getting the image path
        image_name = self.data_dict[idx]
        # Reading the image from the image 
        image = Image.open(image_name)
        image = image.convert('L')
        image = np.array(image)
        image = image.astype(float)
        image = resize(image, (self.image_size, self.image_size))
        # Min-Max normalization
        image = min_max_norm(image)
        # Getting the largest contour
        image = large_contour(image)
        # Resizing the image
        image = resize(image, (self.image_size, self.image_size))

        if self.transform:
           image = self.transform(image)

        return image, idx
    
# Getting the arguments and parsing them
parser = argparse.ArgumentParser()
parser.add_argument("--diff_write_path")
parser.add_argument("--device")
parser.add_argment("--model_weights")
parser.add_argument("--csv_file")

args = parser.parse_args()

# Paramters
imgSize = 512
VAL_BATCH = 1
MODEL_WEIGHTS = args.model_weights
VAL_THRESH_CSV = args.csv_file


val_dict = get_data_inference(VAL_THRESH_CSV)
# Loading the transforms
val_transforms = transforms.Compose([
										transforms.ToTensor(),
										transforms.Resize((imgSize, imgSize))
										# transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
									])

val_transformed_dataset = CTVITDataset(data_dict=val_dict, transform = val_transforms, image_size=imgSize)
# Defining the dataloaders
val_dataloader = DataLoader(dataset = val_transformed_dataset)


# Defining the models
v = ViT(
    image_size = imgSize,
    patch_size = 16,
    channels = 1,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

model = MAE_Recon(
    encoder = v,
    masking_ratio = 0.25,   # the paper recommended 75% masked patches (3 patches 10% masking, 6 patches 20% masking (similar to CNN))
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6,       # anywhere from 1 to 8
    patch_size = 16
)

# Let us come back here after sometime bruh
# Dataparallelize model
model = torch.nn.DataParallel(model)
# # Device and Loading Weights
checkpoint = torch.load(MODEL_WEIGHTS, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
device = f'cuda:{args.device}'
model = model.module.to(device)
model.eval()


for data, idx in tqdm(val_dataloader):
    inp,out = model(data.float().to(device))
    inp = np.squeeze(inp, axis=1)
    out = np.squeeze(out, axis=1)
    
    diff = inp - out
    diff = np.squeeze(diff)

    recon_write_folder = args.recon_write_path
    diff_write_folder = args.diff_write_path

    np.save(os.path.join(diff_write_folder, idx[0]+'.npy'), diff)