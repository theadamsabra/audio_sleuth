import os
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module, DataParallel

class Trainer:
    '''
    Core trainer class. 

    Args:
        model (Module):
        loss (Module):
        optimizer (Optimizer):
        dataloader (DataLoader):
        save_dir (str):
        device (str):
        num_epochs (int):
        parallel (bool):
        device_ids (bool):
    '''
    def __init__(self, model:Module, loss:Module, optimizer:Optimizer, dataloader:DataLoader, \
        save_dir:str, device:str, num_epochs:int, parallel:bool=False, device_ids:list[int]=None) -> None:

        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.dataloader = dataloader 
        self.save_dir = save_dir
        self.device = device
        self.num_epochs = num_epochs
        self.parallel = parallel
        self.device_ids = device_ids

        # Make parallel if defined:
        if self.parallel: 
            assert self.device_ids != None, "Please set device_ids to run training in parallel."
            self.model = DataParallel(self.model, self.device_ids) 

        self.model.to(device)

    def _check_and_create_save_dir(self):
        '''Check if self.save_dir exists and create it if it doesn't.'''
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)
    
    def train(self):
        '''Core train function.'''
        # Create save directory:
        self._check_and_create_save_dir()

        # Main training loop:
        print('BEGINNING TRAINING:')
        for epoch in range(self.num_epochs):
            losses = []
            print(f'TRAINING EPOCH {epoch+1}')
            for audio, labels in self.dataloader:
                audio, labels = audio.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(audio)
                loss_val = self.loss(output, labels)
                loss_val.backward()
                losses.append(loss_val.item())
                self.optimizer.step()

            # Get average epoch:
            average_epoch = sum(losses) / len(losses)
            print(f'AVERAGE LOSS OF EPOCH: {average_epoch}')

            # TODO: save every N checkpoints:
            torch.save(
                {
                    'state_dict': self.model.state_dict()
                },
                os.path.join(self.root_dir, f'checkpoint_{epoch+1}.pth')
            )