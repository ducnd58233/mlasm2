import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import sleep
import os, sys
import numpy as np

filepath = os.getcwd() + '/model_saved/'
DEVICE = torch.device('cuda') \
            if torch.cuda.is_available() \
            else torch.device('cpu')


def _loops(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim,
    loss_criteria: torch.nn,
    num_classes: int,
    mode: str,
    best_val_acc: float = 0.0,
    model_name: str = None
) -> float:
    total_loss = 0
    correct = 0
    batch = 0
    length = len(data_loader.dataset)
    epoch_loss = []
    epoch_acc = []
    test_prediction = []
    
    with tqdm(data_loader, unit = 'batch') as tepoch:
        for data, target in tepoch:
            if mode == 'train':
                tepoch.set_description('Train')
            elif mode == 'val':
                tepoch.set_description('Val')
            else:
                tepoch.set_description('Test')
            batch += 1
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            
            if mode == 'train':
                optimizer.zero_grad() 
                
            out = model(data)
            
            if mode != 'test':
                loss = loss_criteria(out, target)
                total_loss += loss.item()
                epoch_loss.append(total_loss)
            
            if (num_classes == 1):
                _, predicted = torch.max(out.data, 1)
                correct += torch.sum(target == predicted).item()
            else:
                predicted = out.argmax(dim=1)
                correct += torch.sum(target == predicted)
            
            if mode == 'test':
                test_prediction.append(predicted.item())
            
            if mode == 'train':
                loss.backward()
                optimizer.step()
            
            if mode != 'test':              
                acc = 100. * correct.to(DEVICE) / length
                epoch_acc.append(acc)
                guess = f'{correct}/{length}'
                string = f'loss: {loss:.6f}, accuracy: {acc:.6f}% [{guess}]'
                tepoch.set_postfix_str(string)
                
                sleep(0.01)
    
    if mode == 'test':
        return test_prediction    
   
    epoch_loss = epoch_loss[-1]
    epoch_acc = epoch_acc[-1]
    
     ### Saving best model
    if mode == 'val':
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            saved = {'best_val_acc': best_val_acc, 'model_state_dict': model.state_dict()}
            torch.save(saved, filepath + model_name + "_best.pth")
            print(f'New best validation accuracy: {best_val_acc:.6f}, add to {filepath + model_name + ""}_best.pth')

        return epoch_loss, epoch_acc, best_val_acc
   
    
    return epoch_loss, epoch_acc

def train(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim,
    loss_criteria: torch.nn,
    num_classes: int,
) -> float:
    r"""
    Args:
        model (nn.Module): Model use for training
        data_loader (DataLoader): Data loader of training data
        optimizer (torch.optim): Optimizer uses for training
        loss_criteria (torch.nn): Loss function use for training
        num_classes (int): Number of classes (output shape) of dataset
    """

    model.train()

    return _loops(
        model=model,
        data_loader=data_loader,
        optimizer=optimizer,
        loss_criteria=loss_criteria,
        num_classes=num_classes,
        mode='train'
    )

def val(
    model: nn.Module,
    data_loader: DataLoader,
    loss_criteria: torch.nn,
    num_classes: int,
    best_val_acc: float,
    model_name: str = None,
) -> float:
    r"""
    Args:
        model (nn.Module): Model use for validating
        data_loader (DataLoader): Data loader of validating data
        loss_criteria (torch.nn): Loss function use for validating
        num_classes (int): Number of classes (output shape) of dataset
        best_val_acc (float): The current best accuracy of the model
    """

    model.eval()
    avg_loss = 0.0
    avg_acc = 0.0
    
    with torch.no_grad():
        avg_loss, avg_acc, current_best_val_acc = _loops(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            loss_criteria=loss_criteria,
            num_classes=num_classes,
            mode='val',
            best_val_acc=best_val_acc,
            model_name=model_name
        )

    return avg_loss, avg_acc, current_best_val_acc


def test(
    model: nn.Module,
    data_loader: DataLoader,
    loss_criteria: torch.nn,
    num_classes: int,
) -> float:
    r"""
    Args:
        model (nn.Module): Model use for testing
        data_loader (DataLoader): Data loader of testing data
        loss_criteria (torch.nn): Loss function use for testing
        num_classes (int): Number of classes (output shape) of dataset
    """

    model.eval()
    avg_loss = 0.0
    avg_acc = 0.0
    
    with torch.no_grad():
        test_prediction = _loops(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            loss_criteria=loss_criteria,
            num_classes=num_classes,
            mode='test'
        )

    return test_prediction