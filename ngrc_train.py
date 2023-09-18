import warnings

from tensorboard import summary
from torchsummary import summary

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
# from ngrc_tx import NGRC
from ngrc_2d1 import NGRC
#from ngrc_mixer_ridhwan1 import NgrcMixer
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import mnist, CIFAR10
from torch.utils import data
#import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision.models.mobilenet import mobilenet_v2
from prettytable import PrettyTable
#from thop import profile
#from mlpmixer import MLPMixer


np.random.seed(42)

def count_parameters(model):
  table = PrettyTable(["Modules", "Parameters"])
  total_params = 0
  for name, parameter in model.named_parameters():
      if not parameter.requires_grad: continue
      params = parameter.numel()
      table.add_row([name, params])
      total_params+=params
  print(table)
  print(f"Total Trainable Params: {total_params}")
  return total_params

learning_rate = 0.001
batch_size = 100
n_epoches = 2
num_classes = 10

writer = SummaryWriter('/user/arch/shintani/rc_tensorbord')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# data_tf = torchvision.transforms.Compose(
#     [
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.1307,), (0.3081,))
#     ]
# )
# data_path = r'./mnist/'
# train_data = mnist.MNIST(data_path,train=True,transform=data_tf,download=True)
# test_data = mnist.MNIST(data_path,train=False,transform=data_tf,download=True)
# train_loader = data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
# test_loader = data.DataLoader(test_data,batch_size=batch_size)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
data_path = r'./cifar10/'
train_data = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_data = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


# x_test=x_test.astype(np.float16)
# for param in network.params.values():
#     param[...]=param.astype(np.float16)
image_size = 32
patch_size = 8
hidden_dim = 32

#network = NgrcMixer(
#         patch_size = patch_size,
#         tokens_mlp_dim = hidden_dim,
#         channels_mlp_dim = hidden_dim,
#         n_classes = 10,
#         hidden_dim = hidden_dim,
#         n_blocks = 1)

network = NGRC(k=3, skip=2, patch_size=patch_size)
# network = MLPMixer(in_channels=3, dim=512, num_classes=num_classes, patch_size=patch_size, image_size=32, depth=1, token_dim=256,
#                      channel_dim=2048)
#network = mobilenet_v2(pretrained=True)

#network.classifier[1] = torch.nn.Linear(in_features=network.classifier[1].in_features, out_features=10)
network.to(device)
summary(network.to(device), (3, 32, 32),batch_size = 32)
#count_parameters(network)
# input = torch.randn(1, 3, 32, 32)
# flops, params = profile(network, inputs=(input, ))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))

# optimizer = optim.SGD(network.parameters(), lr=learning_rate)
config = {                                                                                                                                                                                                          
    'batch_size': 64,
    'lr': .001,
    'beta1': 0.851436,
    'beta2': 0.999689,
    'amsgrad': True
} 

# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    network.parameters(), 
    lr=config['lr'], 
    betas=(config['beta1'], config['beta2']), 
    amsgrad=config['amsgrad'], 
)
# optimizer = optim.Adam(lr=0.001, beta1=0.9, beta2=0.999, epsilon=None, decay=0.0, amsgrad=False)

loss_func = nn.CrossEntropyLoss()
torch.set_num_threads(1)
tot_time = 0
for epoch in range(n_epoches):
  network.train()
  train_loss = 0
  train_acc = 0
  loop = tqdm(enumerate(train_loader), total=len(train_loader), ascii="░▒█")
  for i, (images, labels) in loop:
    images = images.to(device)
    labels = labels.to(device)

    outputs = network(images)
    loss = loss_func(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _,predictions = outputs.max(1)
    num_correct = predictions.eq(labels.view_as(predictions)).sum().item()
    running_train_acc = float(num_correct) / float(images.shape[0])
    train_loss += loss.item()
    train_acc += num_correct
    loop.set_description(f'Epoch [{epoch+1}/{n_epoches}]')
    # loop.set_postfix(loss = f'{loss.item():.2f}',acc = f'{running_train_acc:.2f}')

  train_loss = train_loss / len(train_loader)
  train_acc = train_acc / len(train_loader.dataset)
  writer.add_scalar('Training loss',train_loss,global_step=epoch)
  writer.add_scalar('Training accuracy',train_acc,global_step=epoch)

  network.eval()
  test_loss = 0
  correct = 0
  test_acc = 0
  start = time.time()
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      
      output = network(data)
      loss = loss_func(output, target)
      
      pred = output.argmax(1)
      correct = pred.eq(target.view_as(pred)).sum().item()
      test_loss += loss.item()
      test_acc += correct
  end = time.time()
  test_loss = test_loss / len(test_loader)
  test_acc = test_acc / len(test_loader.dataset)
  writer.add_scalar('Test loss',test_loss,global_step=epoch)
  writer.add_scalar('Test accuracy',test_acc,global_step=epoch)
  elapsed_time = end - start
  tot_time += elapsed_time
#torch.save(network.state_dict(), '/mnt/sheng/ESN/model/mobile.pt')
FPS = len(test_loader)/(tot_time/n_epoches)
print('FPS=',FPS)
