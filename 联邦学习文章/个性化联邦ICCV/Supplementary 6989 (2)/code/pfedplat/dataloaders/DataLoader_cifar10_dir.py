
import pfedplat as fp
import os
import torch
import torchvision
import random
from torchvision import transforms as transforms
import numpy as np
import copy


class DataLoader_cifar10_dir(fp.DataLoader):
    def __init__(self,
                 pool_size=100,
                 alpha=0.1,
                 batch_size=100,
                 input_require_shape=None,
                 shuffle=True,
                 recreate=False,
                 params=None,
                 *args,
                 **kwargs):

        if params is not None:
            pool_size = params['N']
            alpha = params['Diralpha']
            batch_size = params['B']

        name = 'CIFAR10_dir_pool_' + str(pool_size) + 'alpha_' + str(alpha) + '_batchsize_' + str(
            batch_size) + '_sort_split_input_require_shape_' + str(input_require_shape)
        nickname = 'cifar10 dir B' + \
            str(batch_size) + ' alpha' + str(alpha) + ' N' + str(pool_size)
        super().__init__(name, nickname, pool_size, batch_size, input_require_shape)

        file_path = fp.pool_folder_path + name + '.npy'
        if os.path.exists(file_path) and (recreate == False):
            data_loader = np.load(file_path, allow_pickle=True).item()
            for attr in list(data_loader.__dict__.keys()):
                setattr(self, attr, data_loader.__dict__[attr])
            print('Successfully Read the Data Pool.')
        else:

            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            trainset = torchvision.datasets.CIFAR10(root=fp.data_folder_path, train=True,
                                                    download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=trainset.data.shape[0], shuffle=False, num_workers=1)
            testset = torchvision.datasets.CIFAR10(root=fp.data_folder_path, train=False,
                                                   download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=testset.data.shape[0], shuffle=False, num_workers=1)

            for _, train_data in enumerate(trainloader, 0):
                trainset.data, trainset.targets = train_data
            for _, test_data in enumerate(testloader, 0):
                testset.data, testset.targets = test_data
            dataset_input = []
            dataset_label = []

            dataset_input.extend(trainset.data.cpu().detach().numpy())
            dataset_input.extend(testset.data.cpu().detach().numpy())
            dataset_label.extend(trainset.targets.cpu().detach().numpy())
            dataset_label.extend(testset.targets.cpu().detach().numpy())

            self.cal_data_shape(trainset.data.shape)
            dataset_input = np.array(dataset_input).reshape(
                [-1] + self.input_data_shape)
            dataset_label = np.array(dataset_label)
            self.target_class_num = 10
            self.total_training_number = len(trainset)
            self.total_test_number = len(testset)

            train_prob = trainset.data.shape[0] / \
                (trainset.data.shape[0] + testset.data.shape[0])

            X, y, statistic = fp.separate_data((dataset_input, dataset_label), train_prob, pool_size, self.target_class_num,
                                               item_classes_num=None, batch_size=batch_size, alpha=alpha, niid=True, balance=None, partition='dir')
            self.statistic = statistic

            self.data_pool = fp.create_data_pool(
                X, y, pool_size, shuffle, train_prob, batch_size, self.target_class_num)

            np.save(file_path, self)
