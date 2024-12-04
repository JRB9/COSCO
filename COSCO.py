import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.nn.modules.batchnorm import _BatchNorm
import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

lr = 0.01
rho = 0.1
nEpoch = 100

class PrototypicalLoss:
  # This is just to initilize it
    def __init__(self, flag='neg'):
        self.flag = flag


# The centroid Computing
    def _compute_per_class_centroid(self, i, label, data):
        label1d = label.squeeze()
        data_class = data[label1d == i, :]
        return torch.mean(data_class, 0, True)

    def _compute_class_centroid(self, label, data):
        unique_labels = label.unique().squeeze()
        centroids = self._compute_per_class_centroid(unique_labels[0], label, data)
        for i in range(1, len(unique_labels)):
            index = unique_labels[i]
            centroids = torch.cat((centroids, self._compute_per_class_centroid(index, label, data)), dim=0)
        return centroids

# The similarity or distance computaions
    def _cosine_similarity(self, data, centroid):
        data_norm = torch.nn.functional.normalize(data, dim=1)
        centroid_norm = torch.nn.functional.normalize(centroid, dim=1)
        similarity = torch.mm(data_norm, centroid_norm.t())
        return similarity

    def _similarity_matrix(self, data, centroid):
        epsilon = 1e-6
        similarity = 1 / (torch.cdist(data, centroid, p=2) + epsilon)
        return similarity

    def _distance_matrix(self, data, centroid):
        distance = torch.cdist(data, centroid, p=2)
        return distance
# Using the computations to get loss
    def _prototypical_loss_sim(self, S, labels, alpha=0.01):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(S / alpha)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_neg(self, D, labels):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-D)
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

    def _prototypical_loss_negexp(self, D, labels, alpha = 0.1):
        softmax = torch.nn.Softmax(dim=1)
        o = softmax(-alpha*torch.exp(D))
        labels = labels.squeeze().long()
        loss = F.cross_entropy(o, labels, reduction='mean')
        return loss

#The call function
    def __call__(self, data,label):
        if self.flag == 'neg':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_neg(distance, label)
        elif self.flag == 'sim':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._similarity_matrix(data, centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'cos':
            centroids = self._compute_class_centroid(label, data)
            similarity = self._cosine_similarity(data.detach(), centroids)
            return self._prototypical_loss_sim(similarity, label)
        elif self.flag == 'negexp':
            centroids = self._compute_class_centroid(label, data)
            distance = self._distance_matrix(data, centroids)
            return self._prototypical_loss_negexp(distance, label)

def prototypical_testing(test_embed, train_centroids):

  # Making them both be on Cpu
    test_embed = test_embed.cpu()
    train_centroids = train_centroids.cpu()

# Computing the distabce
    cdist = torch.cdist(test_embed, train_centroids)

#Pick the most similar aka the one with the smallest distance

    test_label = torch.argmin(cdist,dim=1)

    return test_label #the testing label based on training centroid


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

import os
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class ResNetTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X_train, y_train, epochs=500, batch_size=128, eval_batch_size=128):
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        train_dataset = Dataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=4e-3)

        best_acc = 0.0

        # Training
        for epoch in range(epochs):
            model.train()
            for i, inputs in enumerate(train_loader, 0):
                X, y = inputs
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(X)
                loss = F.nll_loss(outputs, y)
                loss.backward()
                optimizer.step()

            model.eval()
            acc = compute_accuracy(model, train_loader, self.device)
            if acc >= best_acc:
                best_acc = acc
                torch.save(model.state_dict(), file_path)
            self.logger.log('--> Epoch {}: loss {:5.4f}; accuracy: {:5.4f}; best accuracy: {:5.4f}'.format(epoch, loss.item(), acc, best_acc))

        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        return model

    def test(self, model, X_test, y_test, batch_size=128):
        test_dataset = Dataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        acc = compute_accuracy(model, test_loader, self.device)
        return acc

def compute_accuracy(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            X, labels = data
            X, labels = X.to(device), labels.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc

class ResNet(nn.Module):

    def __init__(self, input_size, nb_classes):
        super(ResNet, self).__init__()
        n_feature_maps = 128

        self.block_1 = ResNetBlock(input_size, n_feature_maps)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps)
        self.linear = nn.Linear(n_feature_maps, nb_classes)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = F.avg_pool1d(x, x.shape[-1]).view(x.shape[0],-1)
        x2 = self.linear(x)
        y = F.log_softmax(x2, dim=1)

        return y,x

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.expand = True #if in_channels < out_channels else False

        self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)

        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

# mount drive

#from google.colab import drive
#drive.mount('/content/drive')

import numpy as np
import torch
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

def load_data(dataset_name, shot_dir, normalize_data=False):
  data_dir = 'drive/MyDrive/classification_data/'
  #dataset_name = 'PenDigits/'
  #shot_dir = '5-shot/'

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

import torch.optim as optim

def proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size):

  model_resnet = ResNet(input_size = input_size, nb_classes=len(np.unique(train_label)))
  criterion = nn.CrossEntropyLoss()

  # swap run type here
  runSAM = True
  #runSAM = False

  optimizer = 'sgd'
  #optimizer = 'adam'

  criterion = PrototypicalLoss(flag='neg')

  if runSAM==False:
    optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
    #optimizer = optim.Adam(model_resnet.parameters(), lr=1e-8)
  else:
    base_optimizer = torch.optim.SGD # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=rho)

  model_resnet = model_resnet.cuda()

  # SAM

  #100 epoch for batch = 1024*
  best_loss = 10000 #smaller is better.
  max_limit = 20
  counter = 0
  #model_resnet = resnet18()


  #optimizer = optim.Adam(model_resnet.parameters(), lr=1e-3)

  #model_resnet = resnet18().cuda()
  #base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
  #optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=0.1, momentum=0.9)

  #criterion = nn.CrossEntropyLoss()


  for epoch in range(nEpoch):  # loop over the dataset multiple times
      running_loss = 0.0
      val_running_loss = 0.0
      all_embeddings =[]
      all_labels = []
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs = inputs.cuda()
          labels = labels.cuda()
          # print(inputs.shape)
        # print(inputs.shape)

          # first forward-backward step
          enable_running_stats(model_resnet)# <- this is the important line


          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs1 = model_resnet(torch.tensor(inputs).transpose(1,2))
          outputs = outputs1[0]#1
          embed = outputs1[1]
          # print(outputs.shape)

          # print(type(outputs))
          labels = torch.squeeze(labels, dim=1)
          #def closure():
          #  loss = criterion(outputs, labels)
          #  loss.backward()
          #  return loss

          loss = criterion(embed, labels)
          #loss = criterion(outputs, labels
          loss.backward()
          #optimizer.step(closure)
          optimizer.first_step(zero_grad= True)
          # second forward-backward step
          disable_running_stats(model_resnet)  # <- this is the important line
          tmp = criterion(model_resnet(torch.tensor(inputs).transpose(1,2).float())[1], labels)
          tmp.backward()
          optimizer.second_step(zero_grad=True)


          optimizer.zero_grad()

          # print statistics
          running_loss += loss.item()

          # Extracting the embedding and labeles for the 100 epotch
          if epoch ==nEpoch-1:
            all_embeddings.append(embed.detach().cpu())
            all_labels.append(labels.detach().cpu())

      if epoch == nEpoch-1:

          all_embeddings = torch.cat(all_embeddings)
          print(all_embeddings.size())
          all_labels = torch.cat(all_labels)
          train_centroids = criterion._compute_class_centroid(all_labels, all_embeddings)

      print("Epoch:", epoch+1, "-->", running_loss, loss.item(), tmp.item())
          #print("Epoch:", epoch+1, "-->","train loss: ",loss.item(), "second loss: ", tmp.item())

  print('Finished Training')

  torch.save(train_centroids, 'train_centroids.pt')

  #test_data = test_data.cpu().numpy()
  #test_data = np.load(data_dir + dataset_name + 'X_test.npy')
  test_data = torch.from_numpy(test_data).float()
  test_data = test_data.cuda()

  pred, embed = model_resnet(test_data.transpose(1,2).float())
  #pred = model_resnet(test_data.transpose(1,2).float())

  #Loading the saved train_centroids
  train_centroids = torch.load('train_centroids.pt')

  predicted_test_labels =prototypical_testing(embed,train_centroids)
  correct = 0
  total = 0

  labels = torch.squeeze(torch.from_numpy(test_label), dim=1)
  # _, predicted = torch.max(pred.data, 1) Comment this out cause the function gicves us the class
  total = labels.size(0)
  correct = (predicted_test_labels.cuda() == labels.cuda()).sum().item()
  acc = correct/total

  print("Final Accuracy: ",acc)

  return acc

def save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc):
  path = data_dir + dataset_name + shot_dir
  with open(path+'results_sam_proto_neg.txt', 'w') as f:
    f.write(dataset_name + '\n')
    f.write(shot_dir + '\n')
    f.write(str(normalize_data) + '\n')
    f.write(str(acc) + '\n')
    f.write(str(acc.mean()))

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
  data_dir = 'drive/MyDrive/classification_data/'
  #dataset_name = 'ArticularyWordRecognition/'
  #shot_dir = '5-shot/'


  print('reading ' + dataset_name + '...')
  train_data, train_label, test_data, test_label = load_data(dataset_name, shot_dir, normalize_data)
  print(train_data.shape)

  traindata = Dataset(train_data, train_label)

  input_size = train_data.shape[-1]

  print(input_size)

  batch_size = 1024

  trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True, num_workers=2)

  acc = []
  for i in range(5):
    acc_tmp = proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size)
    print(i)
    acc.append(acc_tmp)

  acc = np.array(acc)

  # save the data
  save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc)

  # save to dataframe
  save_to_dataframe(filepath, dataset_name, shot_dir, normalize_data, acc)

  return acc

# Training logic here, replace with desired locations and data.
'''
# columns for our results dataframe
columns = ["Dataset", "Shots", "Normalization", "Result"]

# dataframe construction
df = pd.DataFrame(columns = columns)

# filepath for our csv
filepath = 'drive/MyDrive/classification_data/' + 'training_results.csv'

# creating empty df and csv
df.to_csv(filepath, index=False)
'''

# example with normalization
'''
# NORMALIZE
# change datasets and shot here
dataset_name_vec = ['PEMS-SF/']
shot_dir_vec = ['5-shot/']



#dataset_name = 'ArticularyWordRecognition/'
#shot_dir = '10-shot/'

for dataset_name in dataset_name_vec:
  for shot_dir in shot_dir_vec:
    full_training(dataset_name, shot_dir, True)
'''
