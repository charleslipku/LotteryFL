import numpy as np
import os
import torch
import json
import pandas as pd

class MiniImagenetDataset(torch.utils.data.Dataset):
    def __init__(self, mode = 'train', root = '..', transform = None):
        super(MiniImagenetDataset, self).__init__()
        self.root = root
        self.mode = mode
        self.transforms = transform
        path_image_data = os.path.join(root, 'data_{}.npy'.format(mode))
        path_targets = os.path.join(root, 'targets_{}.npy'.format(mode))
        image_data = np.load(path_image_data)
        targets = np.load(path_targets)
        self.images = image_data
        self.targets = targets


    def __getitem__(self, index):
        return self.transforms(self.images[index]), self.targets[index]

    def __len__(self):
        return self.images.shape[0]



class FemnistDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, transform = None):
        self.data = np.asarray(data_x, dtype=np.float32).reshape(-1, 28, 28, 1)
        self.labels = torch.LongTensor(data_y)
        self.transform = transform
    def __getitem__(self,  index):
        image = self.data[index]
        if(self.transform is not None):
            image = self.transform(image)
        label = self.labels[index]
        return image, label
    def __len__(self):
        return len(self.labels)



def read_data_json(data_dir):
    user_group_train = {}
    user_group_test = {}
    users_list = []
    data_x_train = []
    data_y_train = []
    data_x_test = []
    data_y_test = []
    num_users_train_total = 0
    num_sample_train_total = 0
    num_users_test_total = 0
    num_sample_test_total = 0
    #handle train data first
    files = os.listdir(os.path.join(data_dir, 'train'))
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, 'train' ,f)
        #handle json files
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        #users in this json file
        users = cdata.get('users')
        raw_data = cdata.get('user_data')
        users_list.extend(users)
        for i in range(len(users)):
            #handle each user
            data_x_train.extend(raw_data.get(users[i]).get('x'))
            data_y_train.extend(raw_data.get(users[i]).get('y'))
            user_group_train[num_users_train_total] = [i + num_sample_train_total for i in range(cdata.get('num_samples')[i])]
            assert cdata.get('num_samples')[i] == len(raw_data.get(users[i]).get('x'))
            num_users_train_total += 1
            num_sample_train_total += cdata.get('num_samples')[i]
    #handle test data
    files = os.listdir(os.path.join(data_dir, 'test'))
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, 'test',f)
        #handle json files
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        #users in this json file
        users = cdata.get('users')
        raw_data = cdata.get('user_data')
        for i in range(len(users)):
            #handle each user
            data_x_test.extend(raw_data.get(users[i]).get('x'))
            data_y_test.extend(raw_data.get(users[i]).get('y'))
            user_group_test[users_list.index(users[i])] = [i + num_sample_test_total for i in range(cdata.get('num_samples')[i])]
            assert cdata.get('num_samples')[i] == len(raw_data.get(users[i]).get('x'))
            num_users_test_total += 1
            num_sample_test_total += cdata.get('num_samples')[i]
    
    assert len(user_group_train) == len(user_group_test)
    print("there are {} users\n".format(len(user_group_train)))
    assert len(data_x_train) == len(data_y_train)
    assert len(data_x_test) == len(data_y_test)
    #print(len(user_group_train))
    #print('\n')
    #print(user_group_test)
    return data_x_train, data_y_train, user_group_train, data_x_test, data_y_test, user_group_test

class HARDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, transform = None):
        self.data = torch.from_numpy(data_x).float()
        self.targets = torch.LongTensor(data_y)
        self.transform = transform
    def __getitem__(self,  index):
        feature = self.data[index]
        if(self.transform is not None):
            feature = self.transform(feature)
        label = self.targets[index]
        return feature, label
    def __len__(self):
        return len(self.targets)

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

def read_data_HAR(dataset, num_sample):
    X_train = load_file(os.path.join(dataset, 'train/X_train.txt'))
    print(X_train.shape)
    y_train = load_file(os.path.join(dataset, 'train/y_train.txt'))
    subject_train = load_file(os.path.join(dataset, 'train/subject_train.txt'))
    X_test = load_file(os.path.join(dataset, 'test/X_test.txt'))
    y_test = load_file(os.path.join(dataset, 'test/y_test.txt'))
    subject_test = load_file(os.path.join(dataset, 'test/subject_test.txt'))
    x = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0).squeeze()-1
    subject = np.concatenate([subject_train, subject_test], axis=0).squeeze()
    user_group_train = {}
    user_group_test = {}
    for i in range(30):
        idxs = np.where(subject==(i+1))[0]
        np.random.shuffle(idxs)
        #idxs = idxs[0:num_sample]
        train_idxs = idxs[:num_sample]
        test_idxs = idxs[num_sample:]
        user_group_train[i] = train_idxs
        user_group_test[i] = test_idxs
    return x, y, user_group_train, x, y, user_group_test

def read_data_HAD(data_path, num_sample):
    user_group_train = {}
    user_group_test = {}
    x = np.load(os.path.join(data_path, 'X_new.npy'), allow_pickle=True)
    y = np.load(os.path.join(data_path, 'Y.npy'), allow_pickle=True)
    subject = np.load(os.path.join(data_path, 'Subject.npy'), allow_pickle=True)
    for i in range(14):
        idxs = np.where(subject==(i))[0]
        np.random.shuffle(idxs)
        #idxs = idxs[0:num_sample]
        num_sample = len(idxs)
        train_idxs = idxs[:round(num_sample*0.8)]
        test_idxs = idxs[round(num_sample*0.8):]
        user_group_train[i] = train_idxs
        user_group_test[i] = test_idxs
    return x, y, user_group_train, x, y, user_group_test

class HADDataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y, transform = None):
        self.data = torch.from_numpy(data_x).float()
        self.targets = torch.LongTensor(data_y)
        self.transform = transform
    def __getitem__(self,  index):
        feature = self.data[index]
        if(self.transform is not None):
            feature = self.transform(feature)
        label = self.targets[index]
        return feature, label
    def __len__(self):
        return len(self.targets)
