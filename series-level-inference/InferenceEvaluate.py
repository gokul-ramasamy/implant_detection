# Importing the required packages
import pandas as pd
import nibabel as nib
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt
import cc3d
from tqdm import tqdm
import argparse
import os

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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

def read_volume_and_resize(nifti_path: str, resize_shape: tuple) -> np.array:
    """ Give a NIFTI Path, reads the array,
        converts each slice to windowed slice
        Returns the final array """
    
    arr = nib.load(nifti_path).dataobj
    return_arr = np.zeros((resize_shape[0], resize_shape[1], arr.shape[2]))
    for i in range(arr.shape[2]):
        slice = npy_to_img(arr[:, :, i])
        slice = slice.resize(resize_shape)
        slice = np.array(slice)

        return_arr[:,:,i] = slice

    return return_arr

def cc_analysis(arr: np.array, dusting_threshold: int) -> int:
    connectivity = 26
    # CC3d without dusting
    # labels_out = cc3d.connected_components(arr, connectivity=connectivity)
    # CC3d w dusting
    labels_out = cc3d.dust(arr, threshold=dusting_threshold, connectivity=connectivity, in_place=False)
    x,y,z = labels_out.nonzero()
    # Getting the stats
    stats = cc3d.statistics(labels_out)
    # Getting the voxel count
    count = stats['voxel_counts']

    if count.shape[0] == 2:
        voxel_count = count[1]
    else:
        voxel_count = 0

    return voxel_count


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--diff_write_path")
    parser.add_argument("--csv_file")
    parser.add_argument("--write_path")

    args = parser.parse_args()

    test_csv = args.csv_file
    # Read the CSV files and print the statistics
    test_df = pd.read_csv(test_csv)
    # Let us remove the Unnamed columns
    test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

    for ind in tqdm(test_df.index.values.tolist()):
        nifti_path = test_df.loc[ind, 'ResampledNIFTIPath']
        nifti_ind = nifti_path.split('/')[-1].split('.')[0]
        root_folder = args.diff_write_path

        # Reading and preprocessing the volume
        resize_shape = (256,256) ## Paramter
        arr = np.load(os.path.join(root_folder, nifti_ind+'.npy'))
        # Thresholding the image
        binary_threshold = 0.5 ## Paramter  
        thresholded_arr = arr > binary_threshold
        # 3D Connected component analysis
        dusting_threshold = 100 ## Parameter
        # Getting the voxel count
        voxel_count = cc_analysis(thresholded_arr, dusting_threshold)
        test_df.loc[ind, f'VoxelCount_{resize_shape}_{binary_threshold}_{dusting_threshold}'] = voxel_count

        # Making the predictions
        voxel_threshold = 1173
        if voxel_count >= voxel_threshold:
            test_df.loc[ind, 'Prediction'] = 1
        else:
            test_df.loc[ind, 'Prediction'] = 0

    test_df.to_csv(args.write_path)
