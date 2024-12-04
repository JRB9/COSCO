import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision import datasets, transforms, models
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import uuid
import argparse

from Prototypical_Loss import PrototypicalLoss
from Prototypical_Loss import prototypical_testing as ptest
from Baselines.ResNet import *
from Baselines.TapNet import *
from SAM import SAM

from utils.load_data import *
from utils.save import *
from utils.proto_model import *

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    #Hyperparameters
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--nEpoch', type=int, default=100)

    #Data Loading
    parser.add_argument('--dataset', type=str, default='CharacterTrajectories')
    parser.add_argument('--shot', type=int, default=1 ,choices=[1,10])
    parser.add_argument('--normalize', type=bool, default=False)

    #Baseline Model
    parser.add_argument('--model', type=str, default='resnet',choices=['resnet','tapnet'])

    #SAM
    parser.add_argument('--sam', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='adam',choices=['sgd','adam'])

    #Prototypical Loss
    parser.add_argument('--prototypical_loss', type=bool, default=True)

    #Other Parameters
    parser.add_argument('--prototypical_loss_type',type=str, default='neg',choices=['neg','sim','cos','negexp'])

    #Saving
    parser.add_argument('--save_dir', type=str, default='/content/classification_data/')
    parser.add_argument('--save_name', type=str, default='results.csv')

    args = parser.parse_args()

    # columns for our results dataframe
    columns = ["Dataset", "Shots", "Normalization", "Result"]

    # dataframe construction
    df = pd.DataFrame(columns = columns)

    # filepath for our csv
    filepath = args.save_dir + args.save_name

    # creating empty df and csv
    os.makedirs(os.path.dirname(filepath), exist_ok=True) 
    df.to_csv(filepath, index=False)


    for dataset_name in [args.dataset]:
        for shot_dir in [args.shot]:
            full_training(args)

