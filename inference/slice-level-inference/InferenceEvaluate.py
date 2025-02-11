# Importing the required files
import pandas as pd
import numpy as np
from tqdm import tqdm
import skimage
import os
import argparse
import configparser

def cca(binary):
    # binary = skimage.morphology.remove_small_objects(binary, min_size=50)
    cca_img, count = skimage.measure.label(binary, connectivity=2, return_num=True)
    return cca_img, count

def cca_area(cca_img):
    # Getting the object features of the image
    object_features = skimage.measure.regionprops(cca_img)
    all_areas = [obj["area"] for obj in object_features]
    return sum(all_areas)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="config.ini")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)


    # Reading the csv file
    thresh_val_df = pd.read_csv(config['InferenceEvaluate']['csvfile'])

    root_folder = config['InferenceEvaluate']['diffwritepath']

    a_threshold = 0.6
    thresh = 166

    for ind in tqdm(thresh_val_df.index.values.tolist()):
        path = thresh_val_df.loc[ind, 'Path']
        idx = path.split('/')[-1].split('.')[0]
        
        heatmap_path = os.path.join(root_folder, idx+'.npy')
        diff = np.load(heatmap_path)

        thresholded_diff = diff > a_threshold
        cca_img, count = cca(thresholded_diff)
        area = cca_area(cca_img)

        thresh_val_df.loc[ind, f"Area_{a_threshold}"] = area

        if area >= thresh:
            thresh_val_df.loc[ind, 'Prediction'] = 1
        else:
            thresh_val_df.loc[ind, 'Prediction'] = 0


    thresh_val_df.to_csv(config['InferenceEvaluate']['writepath'])