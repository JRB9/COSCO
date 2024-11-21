import torch
import argparse
import numpy as np
import torch.utils.data
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
  def __init__(self, X_train, y_train):
    # need to convert float64 to float32 else
    # will get the following error
    # RuntimeError: expected scalar type Double but found Float
    self.X = torch.from_numpy(X_train.astype(np.float32))
    # need to convert float64 to Long else
    # will get the following error
    # RuntimeError: expected scalar type Long but found Float
    self.y = torch.from_numpy(y_train).type(torch.LongTensor)
    self.len = self.X.shape[0]

  def __getitem__(self, index):
    return self.X[index], self.y[index]
  def __len__(self):
    return self.len
  
def load_data(args):
    base_path = "./COSCO/Datasets/"
    dataset = args.dataset
    shot = args.shot
    normalize_data = args.normalize
    epsilon = 1e-8

    test_data = np.load(base_path + dataset + '/X_test.npy')
    test_label = np.load(base_path + dataset + '/y_test.npy')
    train_data = np.load(base_path + dataset + '/' + str(shot) + '-shot/X_train.npy')
    train_label = np.load(base_path + dataset + '/' + str(shot) + '-shot/y_train.npy')
    if normalize_data:
        train_data = (train_data - train_data.mean(axis=1)[:, None]) / (train_data.std(axis=1)[:, None] + epsilon)
        test_data = (test_data - test_data.mean(axis=1)[:, None]) / (test_data.std(axis=1)[:, None] + epsilon)

    return train_data, train_label, test_data, test_label
