# Importing the required packages
import pandas as pd
import torch
import os
import datetime
from matplotlib import pyplot as plt
from torchvision import transforms, models
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from VAE.models import VQVAE_BN as VQVAE
from datagen import CustomBatchSampler, get_volume_data_resampled, CTVITVolumeDataset
from train import vit_model_train
from piqa import SSIM
from vit_pytorch import ViT, MAE
import argparse
import configparser
import ast

# Defining the argparser
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str)
args = parser.parse_args()

# Reading the config file
config = configparser.ConfigParser()
config.read(args.config_file)

# GLOBAL VARIABLES
TRAIN_CSV_PATH = config['General']['traincsvpath']
VAL_CSV_PATH = config['General']['valcsvpath']
RESULTS_FOLDER = config['General']['resultsfolder']
LOG_DIR = config['General']['logdir']
MODEL_SAVE_PATH = config['General']['modelsavepath']

# Required variables
NUM_EPOCHS = int(config['Parameters']['numEpochs'])
imgSize = int(config['Parameters']['imgSize'])
TRAIN_BATCH = int(config['Parameters']['trainBatch'])
VAL_BATCH = int(config['Parameters']['valBatch'])
# Defining the devices
DEVICES = ast.literal_eval(config['Parameters']['device'])
lr = float(config['Parameters']['learningrate'])
reg_weight_mse = float(config['Parameters']['reg_weight_mse'])
reg_weight_ssim = float(config['Parameters']['reg_weight_ssim'])


# Create a results folder
if not os.path.isdir(RESULTS_FOLDER):
    os.mkdir(RESULTS_FOLDER)
# Checkpoints write-path
CHECKPOINT_PATH = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'/'  #################
if not os.path.isdir(MODEL_SAVE_PATH+CHECKPOINT_PATH):
    os.mkdir(MODEL_SAVE_PATH+CHECKPOINT_PATH)
#Log directory
if not os.path.isdir(LOG_DIR+CHECKPOINT_PATH):
    os.mkdir(LOG_DIR+CHECKPOINT_PATH)

# Loss Writer
tf_writer = SummaryWriter(LOG_DIR+CHECKPOINT_PATH+'tf')

# Defining the dataset
train_dict = get_volume_data_resampled(TRAIN_CSV_PATH)
val_dict = get_volume_data_resampled(VAL_CSV_PATH)


# Printing the Dataset Statistics
print("\n\n")
print("*** Dataset Statistics ***")
print(">> Number of training examples = {}".format(len(train_dict)))
print(">> Number of valdiation examples = {}".format(len(val_dict)))
print("\n\n")

# Defining the train and validation transforms (Add and remove transforms as required)
train_transforms = transforms.Compose([
										transforms.ToTensor(),
										transforms.Resize((imgSize, imgSize)),
										# transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
										transforms.RandomHorizontalFlip(),
										transforms.RandomVerticalFlip(),
										transforms.RandomRotation(180)
										])
val_transforms = transforms.Compose([
										transforms.ToTensor(),
										transforms.Resize((imgSize, imgSize))
										# transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
									])


train_transformed_dataset = CTVITVolumeDataset(data_dict=train_dict, transform = train_transforms, image_size=imgSize) 
val_transformed_dataset = CTVITVolumeDataset(data_dict=val_dict, transform = val_transforms, image_size=imgSize)

# Batch sampler # The CustomBatchSampler is written to take in a list (instead on iter)
# train_sampler =  CustomBatchSampler(list(train_dict.keys()), batch_size=TRAIN_BATCH, drop_last=True) 
# val_sampler = CustomBatchSampler(list(val_dict.keys()), batch_size=VAL_BATCH, drop_last=True)

# Defining the dataloaders (Add batch_sampler if required)
train_dataloader = DataLoader(dataset = train_transformed_dataset)
val_dataloader = DataLoader(dataset = val_transformed_dataset)

# Crating a dataloader dictionary
dataloaders = dict()
dataloaders['train'] = train_dataloader
dataloaders['val'] = val_dataloader

#use gpu if available
device = torch.device("cuda:"+str(DEVICES[0]) if torch.cuda.is_available() else "cpu")


# Defining the models
v = ViT(
    image_size = imgSize,
    patch_size = 8,
    channels = 1,
    num_classes = 1000,
    dim = 1024,  ## Should this dimension be changed accordingly??
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

model = MAE(
    encoder = v,
    masking_ratio = float(config['Parameters']['maskratio']),   # the paper recommended 75% masked patches (3 patches 10% masking, 6 patches 20% masking (similar to CNN))
    decoder_dim = 512,      # paper showed good results with just 512
    decoder_depth = 6 ,      # anywhere from 1 to 8
    patch_size = int(config['Parameters']['patchsize'])
)

# DataParallel for faster training
model = torch.nn.DataParallel(model, device_ids = DEVICES)
model = model.to(device)
print(">> MODEL LOADED SUCCESSFULLY")

# Defining the optimizers
optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Early Stopping
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(torch.tensor(metrics)):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
es = EarlyStopping(patience=5)

# SSIM
ssim = SSIM(n_channels=1)

# Call the train function
vit_model_train(model, ssim, NUM_EPOCHS, dataloaders, device, optimizer, scheduler, tf_writer, MODEL_SAVE_PATH+CHECKPOINT_PATH, es, reg_weight_mse, reg_weight_ssim)