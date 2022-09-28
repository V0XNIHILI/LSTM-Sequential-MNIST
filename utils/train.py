import os
import json
import pprint
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from utils.contexts import evaluating, training


def train(
    net: nn.Module,
    train_data_loader: DataLoader,
    test_data_loader: DataLoader,
    criterion: nn.Module,
    device: Optional[torch.device] = None
):
    config = wandb.config

    net = net.to(device)

    optimizer = getattr(optim, config['optimizer']['name'])(net.parameters(),
                          lr=config['optimizer']['lr'],
                          momentum=config['optimizer']['momentum'],
                          weight_decay=config['optimizer']['weight_decay'])

    wandb.watch(net,
                criterion=criterion,
                log="all",
                log_freq=5,
                log_graph=(True))

    model_save_dir_path = f"nets/{wandb.run.name}"
    os.makedirs(model_save_dir_path)

    with training(net):
        for epoch_idx in range(config['train']['n_epochs']):
            for batch_idx, (X, y) in enumerate(train_data_loader):
                optimizer.zero_grad()

                inputs = X.transpose(1, 0).to(device)
                labels = y.to(device)

                output = net(inputs)

                # Taking [0] as output is of shape  (1, batch_size, hidden_size)
                loss = criterion(output[0], labels)

                loss.backward()

                nn.utils.clip_grad_value_(
                    net.parameters(), clip_value=config['optimizer']['grad_clip_value'])

                optimizer.step()

                logging_data = {
                    "loss/train":
                    loss,
                    "average_sequence_loss/train":
                    loss,
                    "epoch":
                    epoch_idx,
                    "batch":
                    batch_idx,
                    "train_step":
                    config['task']['T'] * (batch_idx + 1) +
                    epoch_idx * len(train_data_loader) * config['task']['T'] - 1
                }

                pprint.pprint(logging_data)

                wandb.log(logging_data)

            with evaluating(net), torch.no_grad():
                number_correct = 0

                epoch_test_loss = 0.0

                for _, (X, y) in enumerate(test_data_loader):
                    inputs = X.transpose(1, 0).to(device)
                    labels = y.to(device)

                    output = net(inputs)

                    epoch_test_loss += criterion(output, labels)

                    number_correct += (torch.argmax(
                        output, dim=1) == labels).sum().item()

                logging_data = {
                    "average_epoch_loss/test":
                    epoch_test_loss / len(test_data_loader),
                    "epoch":
                    epoch_idx,
                    "accuracy/test":
                    number_correct / (len(test_data_loader) * config['test']['batch_size'])
                }
                
                pprint.pprint(logging_data)

                wandb.log(logging_data)

                torch.save(net.state_dict(),
                           f"{model_save_dir_path}/epoch_{epoch_idx}.pth")
