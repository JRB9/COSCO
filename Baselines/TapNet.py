# TapNet is from the AEON library: 
#!pip install -U aeon
#!pip install keras-self-attention

import numpy as np
import pandas as pd
from aeon.classification.deep_learning import TapNetClassifier
from aeon.datasets import load_unit_test
from sklearn.metrics import accuracy_score

def train_tapnet(train_data, train_label, test_data, test_label, input_size):
  tapnet = TapNetClassifier(n_epochs=nEpoch,batch_size=8)
  tapnet.fit(train_data, train_label)
  y_pred = tapnet.predict(test_data)

  acc = accuracy_score(test_label, y_pred)
  print("Final Accuracy: ",acc)
  return acc

import torch

def load_data(dataset_name, shot_dir, normalize_data=False):
  data_dir = 'drive/MyDrive/classification_data/'

  #tiny epsilon value
  epsilon = 1e-8

  train_data = np.load(data_dir + dataset_name + shot_dir + 'X_train.npy')
  train_label = np.load(data_dir + dataset_name + shot_dir + 'y_train.npy')
  test_data = np.load(data_dir + dataset_name + 'X_test.npy')
  test_label = np.load(data_dir + dataset_name + 'y_test.npy')
  print(torch.any(torch.isnan(torch.from_numpy(train_data))))
  print(torch.any(torch.isnan(torch.from_numpy(test_data))))
  if normalize_data:
    train_data = (train_data - train_data.mean(axis=1)[:, None]) / (train_data.std(axis=1)[:, None] + epsilon)
    test_data = (test_data - test_data.mean(axis=1)[:, None]) / (test_data.std(axis=1)[:, None] + epsilon)
    print(torch.any(torch.isnan(torch.from_numpy(train_data))))
    print(torch.any(torch.isnan(torch.from_numpy(test_data))))

  return train_data, train_label, test_data, test_label

import pandas as pd

def save_to_dataframe(dataframe_name, dataset_name, shot_dir, normalize_data, acc):
  # path to the dataframe csv file
  path = dataframe_name

  # new dataframe for appending (accuracy tensor in mean)
  new_data = {
      'Dataset': [dataset_name],
      'Shots': [shot_dir],
      'Normalization': [normalize_data],
      'Result': [acc.mean()]
  }

  new_df = pd.DataFrame(new_data)

  # append the new dataframe to the existing csv
  new_df.to_csv(path, mode='a', header=False, index=False)


def full_training(dataset_name, shot_dir, normalize_data = True):
  # default data directory
  data_dir = 'drive/MyDrive/classification_data/'

  # data load
  print('reading ' + dataset_name + '...')
  train_data, train_label, test_data, test_label = load_data(dataset_name, shot_dir, normalize_data)
  print(train_data.shape)

  # dimension size
  input_size = train_data.shape[-1]
  print(input_size)

  # make labels 1 dimensional
  train_label = train_label.reshape(-1)
  test_label = test_label.reshape(-1)

  # accuracy list for the 5 runs
  acc = []
  for i in range(5):
    acc_tmp = train_tapnet(train_data, train_label, test_data, test_label, input_size)
    print(i)
    acc.append(acc_tmp)

  # convert to array
  acc = np.array(acc)

  # save the data
  save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc)

  # save to dataframe
  save_to_dataframe(filepath, dataset_name, shot_dir, normalize_data, acc)

  return acc

from google.colab import drive
drive.mount('/content/drive')

filepath = 'drive/MyDrive/classification_data/' + 'tapnet_results.csv'


# example training logic with normalization.
'''
# NORMALIZE
# change datasets and shot here
dataset_name_vec = ['ArticularyWordRecognition/', 'BasicMotions/', 'CharacterTrajectories/',
  'Epilepsy/', 'EthanolConcentration/', 'FaceDetection/', 'FingerMovements/', 'HandMovementDirection/',
  'Libras/', 'MotorImagery/', 'NATOPS/', 'PEMS-SF/', 'PenDigits/', 'RacketSports/', 'SelfRegulationSCP1/',
  'SelfRegulationSCP2/', 'UWaveGestureLibrary/']
shot_dir_vec = ['1-shot/']



for dataset_name in dataset_name_vec:
  for shot_dir in shot_dir_vec:
    full_training(dataset_name, shot_dir, True)
'''
