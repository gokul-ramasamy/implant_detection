import SimpleITK as sitk
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse


### Function to resample the volume
def sitk_resample(itk_image, out_spacing=[1.0,1.0,1.0], interpolation=None):
    # Getting the original attributes
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    original_origin = itk_image.GetOrigin()

    resample = sitk.ResampleImageFilter()
    # Setting the output size
    out_size = [
                int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
                ]
    resample.SetSize(out_size)
    #  Setting the output spacing
    resample.SetOutputSpacing(out_spacing)
    # Setting the output direction
    resample.SetOutputDirection(itk_image.GetDirection())
    # Setting the output origin
    resample.SetOutputOrigin(itk_image.GetOrigin())
    # Setting the transform
    resample.SetTransform(sitk.Transform())
    # Setting the default pixel value
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    # Setting the interpolation
    if interpolation == None:
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'Linear':
        resample.SetInterpolator(sitk.sitkLinear)
    elif interpolation == 'NearestNeighbor':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == 'BSpline':
        resample.SetInterpolator(sitk.sitkBSpline)
    else:
        raise Exception("The interpolator should be one of 'Linear', 'NearestNeighbor', 'BSpline'")
    
    return resample.Execute(itk_image)


def registration(df, root_folder, csv_path):
    for ind in tqdm(df.index.values.tolist()):
        path = df.loc[ind, 'NIFTIPath']
        # thickness = df.loc[ind, 'SliceThickness']
        # Reading the volume
        image = sitk.ReadImage(path)
        # Adding the pixel spacing as well
        pixel_spacing_x = image.GetSpacing()[0]
        pixel_spacing_y = image.GetSpacing()[1]
        pixel_spacing_z = image.GetSpacing()[2]
        df.loc[ind, 'PixelSpacing_X'] = pixel_spacing_x
        df.loc[ind, 'PixelSpacing_Y'] = pixel_spacing_y
        df.loc[ind, 'PixelSpacing_Z'] = pixel_spacing_z
        # Resampling the volume
        resampled_image = sitk_resample(image, out_spacing=[1.0,1.0,3.0], interpolation='Linear')

        # Getting the path for resampling the image
        write_path = path.split('/')[-1]
        write_path = os.path.join(root_folder, write_path)
        # Adding the path to the dataframe
        # df.loc[ind, 'NIFTIPath'] = path
        df.loc[ind, 'ResampledNIFTIPath'] = write_path
        df.loc[ind, 'ResampledNIFTISize'] = str(resampled_image.GetSize())

        # Writing the resampled volume to the path
        writer = sitk.ImageFileWriter()
        writer.SetFileName(write_path)
        writer.Execute(resampled_image)


    df.to_csv(csv_path, index=None)

    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv_path")
    parser.add_argument("--output_csv_path")
    parser.add_arguement("--nifti_write_path")

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv_path)
    csv_path = args.output_csv_path
    root_folder = args.nifti_write_path

    registration(df, root_folder, csv_path)
