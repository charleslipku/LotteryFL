#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFeMnist, CNNFeMnist_sim, CNNMiniImagenet, MLP_general
from utils import get_dataset, get_dataset_femnist, get_dataset_cifar10_extr_noniid, get_dataset_HAR, get_dataset_HAD, get_dataset_mnist_extr_noniid, average_weights, average_weights_with_masks, exp_details, make_mask, prune_by_percentile, mask_model, mix_global_weights, get_dataset_miniimagenet_extr_noniid


if __name__ == '__main__':
    start_time = time.time()


    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu:
    #    torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    print('loading dataset: {}...\n'.format(args.dataset))
    if args.dataset == 'femnist':
        data_dir = '/home/leaf/data/femnist/data/' # put your leaf project path here
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_femnist(data_dir)
    elif args.dataset == 'cifar10_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_cifar10_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'miniimagenet_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_miniimagenet_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'mnist_extr_noniid':
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_mnist_extr_noniid(args.num_users, args.nclass, args.nsamples, args.rate_unbalance)
    elif args.dataset == 'HAR':
        data_dir = '../data/UCI HAR Dataset'
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_HAR(data_dir, args.num_samples)
    elif args.dataset == 'HAD':
        data_dir = '../data/USC_HAD'
        train_dataset, test_dataset, user_groups_train, user_groups_test = get_dataset_HAD(data_dir, args.num_samples)
    else:
        train_dataset, test_dataset, user_groups = get_dataset(args)
    print('data loaded\n')


    print('building model...\n')
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist' or args.dataset == 'mnist_extr_noniid':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar' or args.dataset == 'cifar10_extr_noniid':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'femnist':
            global_model = CNNFeMnist_sim(args=args)
        elif args.dataset == 'miniimagenet_extr_noniid':
            global_model = CNNMiniImagenet(args=args)

    elif args.model == 'mlp_general':
        global_model = MLP_general(args)

    else:
        exit('Error: unrecognized model')
    print('model built\n')


    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    init_weights = copy.deepcopy(global_model.state_dict())

    #make masks
    masks = []
    init_mask = make_mask(global_model)
    for i in range(args.num_users):
        if i%100 == 0:
            print('making mask...{}/args.num_users'.format(i))
        masks.append(copy.deepcopy(init_mask))
   

    #list to document the pruning rate of each local model
    pruning_rate = []
    for i in range(args.num_users):
        pruning_rate.append(1)

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    best_acc = [0 for i in range(args.num_users)]
   
   
    for epoch in tqdm(range(args.epochs)):
        users_in_epoch = []
        local_weights, local_losses , local_masks, local_prune = [], [], [], []
        local_acc = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        #sample users for training
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #train local models
        for idx in idxs_users:
            if args.dataset == 'femnist' or args.dataset == 'HAR' or args.dataset == 'HAD' or args.dataset == 'cifar10_extr_noniid' or args.dataset == 'miniimagenet_extr_noniid' or args.dataset == 'mnist_extr_noniid':
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups_train[idx], logger=logger, dataset_test=test_dataset, idxs_test=user_groups_test[idx])
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
            #test global model before train
            train_model = copy.deepcopy(global_model)
            #mask the model
            mask_model(train_model, masks[idx], train_model.state_dict())
            acc_beforeTrain, _ = local_model.inference(model = train_model)
            #if test acc is not bad, prune it
            if(acc_beforeTrain > args.prune_start_acc and pruning_rate[idx] > args.prune_end_rate):
                #prune it
                prune_by_percentile(train_model, masks[idx], args.prune_percent)
                #update pruning rate
                pruning_rate[idx] = pruning_rate[idx] * (1 - args.prune_percent/100)
                #reset to initial value to make lottery tickets
                mask_model(train_model, masks[idx], init_weights)
            #train and get new weights
            w, loss = local_model.update_weights(
                model=train_model, global_round=epoch, device = device)
            #model used for test
            temp_model = copy.deepcopy(global_model)
            temp_model.load_state_dict(w)
            mask_model(temp_model, masks[idx], temp_model.state_dict())
            acc, _ = local_model.inference(model = temp_model)
            print("user {} with {} samples is trained, acc_beforeTrain = {}, acc = {}, loss = {}, parameter pruned = {}%".format(idx, len(user_groups_train[idx]), acc_beforeTrain, acc, loss, (1 - pruning_rate[idx]) * 100))
            #print("user {} is trained, acc_beforeTrain = {}, acc = {}, loss = {}, parameter pruned = {}%".format(idx, acc_beforeTrain, acc, loss, (1 - pruning_rate[idx]) * 100))
            if(args.prune_percent != 0):
                users_in_epoch.append(idx)
                if(acc > best_acc[idx]):
                    best_acc[idx] = acc
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_masks.append(copy.deepcopy(masks[idx]))
            local_prune.append(pruning_rate[idx])
            #if acc < 0.95:
            local_acc.append(acc)
        print("local accuracy: {}\n".format(sum(local_acc)/len(local_acc)))
        
        # update global weights
        #global_weights = average_weights(local_weights)
        global_weights_epoch = average_weights_with_masks(local_weights, local_masks, device)
        global_weights = mix_global_weights(global_weights, global_weights_epoch, local_masks, device)

        # updatc global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        #compute communication cost rate in this epoch
        communication_cost_epoch = sum(local_prune) / len(local_prune)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        #if the users amount is too large, random choose some for test
        if args.num_users > 500:
            users_for_test = np.random.choice(args.num_users, 500, replace=False)
        else:
            users_for_test = range(args.num_users)
        for idx in users_for_test:
            if args.dataset == 'femnist' or args.dataset == 'HAR' or args.dataset == 'HAD' or args.dataset == 'cifar10_extr_noniid' or args.dataset == 'miniimagenet_extr_noniid' or args.dataset == 'mnist_extr_noniid':
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups_train[idx], logger=logger, dataset_test=test_dataset, idxs_test=user_groups_test[idx])
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
            local_mask_model = copy.deepcopy(global_model)
            mask_model(local_mask_model, masks[idx], local_mask_model.state_dict())
            acc, loss = local_model.inference(model=local_mask_model)
            if(args.prune_percent != 0):
                users_in_epoch.append(idx)
                if(acc > best_acc[idx]):
                    best_acc[idx] = acc
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            if(args.prune_percent != 0):
                best_acc_sum = 0
                for i in users_in_epoch:
                    best_acc_sum += best_acc[i]
                best_acc_avg = best_acc_sum/len(users_in_epoch)
            else:
                best_acc_avg = train_accuracy[-1]
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Test Accuracy: {:.2f}% \n'.format(100*best_acc_avg))
            print('Percent of Communication Saved in this Round: {:.2f}% \n'.format(100*(1-communication_cost_epoch)))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    

