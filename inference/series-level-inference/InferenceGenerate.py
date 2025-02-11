# Importing the required packages
import pandas as pd
import torch
import os
import datetime
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import resize

import torch.nn.functional as F
from VAE.models import VQVAE_BN as VQVAE
from piqa import SSIM
from vit_pytorch import ViT, MAE_Recon
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import cv2
import configparser

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",default="config.ini")
args = parser.parser_args()

config = configparser.ConfigParser()
config.read(args.config_file)


# Parameters
imgSize = 256
VAL_BATCH = 1
MODEL_WEIGHTS = config['InferenceGenerate']['modelweights']

# Getting the CSV files
test_csv = config['InferenceGenerate']['csvfile']
# Read the CSV file
test_df = pd.read_csv(test_csv)
# Let us remove the Unnamed columns
df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]


# Defining the models
v = ViT(
    image_size = imgSize,
    patch_size = 8,
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
    decoder_depth = 6       # anywhere from 1 to 8
)

# Let us come back here after sometime bruh
# Dataparallelize model
model = torch.nn.DataParallel(model)
# # Device and Loading Weights
checkpoint = torch.load(MODEL_WEIGHTS, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
device = f'cuda:{int(config['InferenceGenerate']['device'])}'
model = model.module.to(device)
model.eval()

# Function to remove the zero pad
def remove_zero_pad(image):
    dummy = np.argwhere(image != 0) # assume blackground is zero
    max_y = dummy[:, 0].max()
    min_y = dummy[:, 0].min()
    min_x = dummy[:, 1].min()
    max_x = dummy[:, 1].max()
    crop_image = image[min_y:max_y, min_x:max_x]

    return crop_image

# Add the volume window stuff here
def vol_window(vol, level, window):
    maxval = level + window/2
    minval = level - window/2
    vol[vol<minval] = minval
    vol[vol>maxval] = maxval
    return vol

# Converting the array to slice image
def npy_to_img(axial_npy):
    axial_npy = np.asarray(axial_npy)
    # Converting the axial npy to png image
    axial_npy = axial_npy.astype(float)
    axial_img = vol_window(axial_npy, 500, 1500)
    axial_img = (axial_img-np.min(axial_img))/(np.max(axial_img)-np.min(axial_img))
    axial_img = 255.0*axial_img
    axial_img = Image.fromarray(np.uint8(axial_img))
    
    return axial_img

# Have to create a Dataset and DataLoader specifically to run this evaluation
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
    return norm

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
    
    cropped_image = image[y:y+h, x:x+w]


    return cropped_image

    
class CTDataset(Dataset):
    def __init__(self, arr, transform):
        self.arr = arr
        self.transform = transform
        self.image_size=imgSize

    # Overriding the __len__ method as the number of images in the root folder
    def __len__(self):
        return self.arr.shape[2]
    
    # Overriding the __getitem__ method to get the images in the root folder
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Use volume window to get the particular image
        image = npy_to_img(self.arr[:, :, idx])
        # Converting the image to grayscale
        image = image.convert('L')
        image = np.array(image)
        image = image.astype(float)
        # Min-max normalization
        image = min_max_norm(image)
        # Resizing the image
        image = resize(image, (self.image_size, self.image_size))
        # Getting the largest contour
        image = large_contour(image)
        # Resizing the image
        image = resize(image, (self.image_size, self.image_size))


        if self.transform:
           image = self.transform(image)

        return image, idx

def fn(im, resize_shape, model):
    """Takes in one image and gives out the reconstruction of masked encoder of that image"""
    
    # masked_imgs = list()
    # masked_imgs.append(im)
    # Getting the dataset and the DataLoader ready
    data_transform = transforms.Compose([
								transforms.ToTensor(),
								transforms.Resize(resize_shape)]) 
    batch_size = 1

    dataset = CTDataset(im, transform = data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Running the model
    reconstructions = list()
    inputs = list()
    indexes = list()
    for i in dataloader:
        image = i[0].float().to(device)
        masked_image, output_image = model(image)
        output_image = output_image.squeeze()

        inputs.append(i[0].cpu().numpy().squeeze())
        reconstructions.append(output_image)

    return reconstructions, inputs, indexes


for ind in tqdm(df.index.values.tolist()):
    nifti_path = df.loc[ind, 'ResampledNIFTIPath']
    
    nifti = nib.load(nifti_path).dataobj
    write_index = nifti_path.split('/')[-1].split('.')[0]

    all_reconstructions = list()
    all_inputs = list()

    for i in range(nifti.shape[2]):
        arr = nifti[:,:,i]
        if len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=2)
        reconstructions, inputs, indexes = fn(arr, (imgSize,imgSize), model)

        all_reconstructions.append(np.expand_dims(reconstructions[0], axis=0))
        all_inputs.append(np.expand_dims(inputs[0], axis=0))

    final_inputs = np.concatenate(all_inputs, axis=0)
    final_reconstructions = np.concatenate(all_reconstructions, axis=0)
    final_reconstructions = final_reconstructions.clip(0)

    final_diff = final_inputs - final_reconstructions


    diff_write_path = os.path.join(config['InferenceGenerate']['diffwritepath'], write_index+'.npy')
    np.save(diff_write_path, final_diff)

    recon_write_path = os.path.join(config['InferenceGenerate']['reconwritepath'], write_index+'.npy')
    np.save(recon_write_path, final_reconstructions)
