import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from COSCO.Prototypical_Loss import PrototypicalLoss
from COSCO.Prototypical_Loss import prototypical_testing as ptest
from COSCO.Baselines.ResNet import *
from COSCO.SAM import SAM

from COSCO.utils.load_data import *
from COSCO.utils.save import *

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


def proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size,args):
  model_resnet = ResNet(input_size = input_size, nb_classes=len(np.unique(train_label)))
  criterion = nn.CrossEntropyLoss()
  lr = args.lr
  rho = args.rho
  nEpoch = args.nEpoch

  runSAM = args.sam
  optimizer = args.optimizer

  criterion = PrototypicalLoss(flag='neg')

  if runSAM==False:
    optimizer = torch.optim.SGD(model_resnet.parameters(), lr=lr, momentum=0.9)
  else:
    base_optimizer = torch.optim.SGD # define an optimizer for the "sharpness-aware" update
    optimizer = SAM(model_resnet.parameters(), base_optimizer, lr=lr, momentum=0.9, rho=rho)

  model_resnet = model_resnet.cuda()

  best_loss = 10000
  max_limit = 20
  counter = 0

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

  predicted_test_labels =ptest(embed,train_centroids)
  correct = 0
  total = 0

  labels = torch.squeeze(torch.from_numpy(test_label), dim=1)
  # _, predicted = torch.max(pred.data, 1) Comment this out cause the function gicves us the class
  total = labels.size(0)
  correct = (predicted_test_labels.cuda() == labels.cuda()).sum().item()
  acc = correct/total

  print("Final Accuracy: ",acc)

  return acc

def full_training(args):
  train_data, train_label, test_data, test_label = load_data(args)

  traindata = Dataset(train_data ,train_label)

  input_size = train_data.shape[-1]

  if args.model == "tapnet":
     test_label =test_label.reshape(-1)
     train_label = train_label.reshape(-1)

  elif args.model=="resnet":
    batch_size = 1024
    trainloader = DataLoader(traindata, batch_size=batch_size,
                         shuffle=True, num_workers=2)
  acc = []
  for i in range(5):
    if args.model=='tapnet':
        acc_tmp = train_tapnet(train_data, train_label, test_data, test_label, input_size)
    elif args.model =='resnet':
       acc_tmp = proto_neg_train_model(trainloader, train_label, test_data, test_label, input_size,args)
    print(i)
    acc.append(acc_tmp)

  acc = np.array(acc)

  # save the data
  save_to_file_directory(acc,args)

  # save to dataframe
  save_to_dataframe(acc,args)

  return acc
