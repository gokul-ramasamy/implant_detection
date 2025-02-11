import pandas as pd
from torch.utils.data import Dataset, BatchSampler
import torch
from skimage import io
from skimage.transform import resize
import numpy as np
import copy
from PIL import Image
import random
import cv2
from matplotlib import pyplot as plt
from typing import TYPE_CHECKING
import nibabel as nib


# Function for performing min-max normalization
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

# Pass in path to csv file after resampling
def get_volume_data_resampled(csv_path):
    df = pd.read_csv(csv_path)
    data_dict = dict()
    df = df[df['SliceThickness'] <= 3.0]
    for i in df.index.values.tolist():
        path = df.loc[i, "ResampledNIFTIPath"]
        # The index is assumed to be the filename without the extension (i.e., without .nii, .ngz)
        ind = path.split("/")[-1].split('.')[0]
        data_dict[ind] = path
    return data_dict


class CTVITVolumeDataset(Dataset):
    def __init__(self, data_dict, transform, image_size):
        self.data_dict = data_dict
        self.transform = transform
        self.image_size = image_size
    
    # Overriding the __len__ methods as the number of images in the root folder
    def __len__(self):
        return len(self.data_dict)
    
    # Overriding the __getitem__ method to get the images in the root folder
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Getting the NIFTPath
        nifti_path = self.data_dict[idx]
        # Getting the volume array from nifti
        nifti = nib.load(nifti_path)
        arr = nifti.dataobj
        # Randomly selecting an image
        num_frames = arr.shape[2]
        frame = np.random.randint(0,num_frames)
        # Use volume window to get the particular image
        image = npy_to_img(arr[:, :, frame])
        # Converting the image to grayscale
        image = image.convert('L')
        image = np.array(image)
        image = image.astype(float)
        # Min-max normalization
        image = min_max_norm(image)
        # Removing the zero-pad
        image = remove_zero_pad(image)
        # Resizing the image
        image = resize(image, (self.image_size, self.image_size))

        if self.transform:
           image = self.transform(image)

        return image, idx

# CustomBatchSampler makes sure no two slices are from the same study in the same batch
class CustomBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last=True, do_random=False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random = do_random
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler)+self.batch_size -1)//self.batch_size
    

    def __iter__(self):
        # How do you write this iter??
        self.sampler = list(self.sampler)
        if self.random:
            random.shuffle(self.sampler)
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = list()
                    ct = list()
                    while len(batch) != self.batch_size:
                        current_one = next(sampler_iter)
                        current_ct = current_one.split('_')[0]
                        
                        if current_ct not in ct:
                            batch.append(current_one)
                            ct.append(current_ct)
                        else:
                            continue

                    yield batch

                except StopIteration:
                    break