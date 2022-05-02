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
from models import MLP, MLP_general, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFeMnist, CNNFeMnist_sim, CNNMiniImagenet
from utils import get_dataset, get_dataset_femnist, get_dataset_cifar10_extr_noniid, get_dataset_HAR, get_dataset_HAD, get_dataset_mnist_extr_noniid, average_weights, average_weights_with_masks, exp_details, make_mask, prune_by_percentile, mask_model, mix_global_weights, get_dataset_miniimagenet_extr_noniid


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    print('loading dataset: {}...\n'.format(args.dataset))
    if args.dataset == 'femnist':
        #data_dir = '/home/js905/code/femnist_data'
        data_dir = '/home/js905/code/leaf/data/femnist/data/'
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

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            if args.dataset == 'HAR' or args.dataset == 'shakespeare' or 'extr_noniid' in args.dataset:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups_train[idx], logger=logger, dataset_test=test_dataset, idxs_test=user_groups_test[idx])
            else:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, device=device)
            # get new model
            new_model = copy.deepcopy(global_model)
            new_model.load_state_dict(w)
           
            print('user {}, loss {}'.format(idx, loss))
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
       
        

        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)


        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print('Global model Accuracy: {:.2f}% \n'.format(100*test_acc))

    #f_rs_new.close()
    #f_rs_old.close()
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

       
