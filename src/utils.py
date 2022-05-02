#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import heapq
import torch
import numpy as np
from dataset import FemnistDataset, read_data_json, MiniImagenetDataset, HARDataset, read_data_HAR, HADDataset, read_data_HAD
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, cifar_extr_noniid, miniimagenet_extr_noniid, mnist_extr_noniid

def get_dataset_femnist(data_dir):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_json(data_dir)
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = FemnistDataset(data_x_train, data_y_train, apply_transform)
    dataset_test = FemnistDataset(data_x_test, data_y_test, apply_transform)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_HAR(data_dir, num_samples):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_HAR(data_dir, num_samples)
    dataset_train = HARDataset(data_x_train, data_y_train)
    dataset_test = HARDataset(data_x_test, data_y_test)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_HAD(data_dir, num_samples):
    data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test = read_data_HAD(data_dir, num_samples)
    dataset_train = HADDataset(data_x_train, data_y_train)
    dataset_test = HADDataset(data_x_test, data_y_test)
    return dataset_train, dataset_test, user_group_train, user_group_test

def get_dataset_cifar10_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../data/cifar/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = cifar_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset_mnist_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../data/mnist/'
    apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = mnist_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset_miniimagenet_extr_noniid(num_users, n_class, nsamples, rate_unbalance):
    data_dir = '../dataset/mini-imagenet/'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    train_dataset = MiniImagenetDataset(mode = 'train', root = data_dir,
                                   transform=apply_transform)

    test_dataset = MiniImagenetDataset(mode = 'test', root = data_dir,
                                      transform=apply_transform)

    # Chose euqal splits for every user
    user_groups_train, user_groups_test = miniimagenet_extr_noniid(train_dataset, test_dataset, num_users, n_class, nsamples, rate_unbalance)
    return train_dataset, test_dataset, user_groups_train, user_groups_test

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fashion_mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        if args.dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)
        else:
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)

        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)


    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_with_masks(w, masks, device):
    '''
    Returns the average of the weights computed with masks.
    '''
    step = 0
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if 'weight' in key:
            mask = masks[0][step]
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
                mask += masks[i][step]
            w_avg[key] = torch.from_numpy(np.where(mask<1, 0, w_avg[key].cpu().numpy()/mask)).to(device)
            step += 1
        else:
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def difference_weights(w_new, w):
    w_diff = copy.deepcopy(w)
    for key in w.keys():
        w_diff[key] = w_new[key] - w[key]
    return w_diff

def cosine_weights(w1, w2):
    cosines = []
    for key in w1.keys():
        cosines.append(torch.cosine_similarity(w1[key].view(-1), w2[key].view(-1), 0))
    return sum(cosines)/len(cosines)


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step = step + 1
    return mask

def make_mask_ratio(model, ratio):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step = step + 1
    mask = [None]* step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            temp_tensor = np.ones_like(tensor)
            temp_array = temp_tensor.reshape(-1)
            zero_index = np.random.choice(range(temp_array.size), size = int(temp_array.size * ratio), replace = False)
            temp_array[zero_index] = 0
            temp_tensor = temp_array.reshape(temp_tensor.shape)
            mask[step] = temp_tensor
            step = step + 1
    return mask

# Prune by Percentile module
def prune_by_percentile(model, mask, percent, resample=False, reinit=False,**kwargs):

        # Calculate percentile value
        step = 0
        for name, param in model.named_parameters():

            # We do not prune bias term
            if 'weight' in name:
                tensor = param.data.cpu().numpy()
                alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
                percentile_value = np.percentile(abs(alive), percent)

                # Convert Tensors to numpy and calculate
                weight_dev = param.device
                new_mask = np.where(abs(tensor) < percentile_value, 0, mask[step])

                # Apply new weight and mask
                param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
                mask[step] = new_mask
                step += 1

# Mask the model
def mask_model(model, mask, initial_state_dict):
    step = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_dev = param.device
            param.data = torch.from_numpy(mask[step] * initial_state_dict[name].cpu().numpy()).to(weight_dev)
            step = step + 1
        if "bias" in name:
            param.data = initial_state_dict[name]

# mix the global_weight_epoch and global_weight
def mix_global_weights(global_weights_last, global_weights_epoch, masks, device):
    step = 0
    global_weights = copy.deepcopy(global_weights_epoch)
    for key in global_weights.keys():
        if 'weight' in key:
            mask = masks[0][step]
            for i in range(1, len(masks)):
                mask += masks[i][step]
            global_weights[key] = torch.from_numpy(np.where(mask<1, global_weights_last[key].cpu(), global_weights_epoch[key].cpu())).to(device)
            step += 1
    return global_weights
