
from pfedplat.main import initialize, read_params, outFunc
import os


from pfedplat.Algorithm import Algorithm
from pfedplat.Client import Client
from pfedplat.DataLoader import DataLoader
from pfedplat.Model import Model
from pfedplat.Metric import Metric
from pfedplat.seed import setup_seed

from pfedplat.metric.Correct import Correct
from pfedplat.metric.Precision import Precision
from pfedplat.metric.Recall import Recall

from pfedplat.model.LeNet5 import LeNet5
from pfedplat.model.CNN import CNN
from pfedplat.model.MLP import MLP
from pfedplat.model.NFResNet import NFResNet18, NFResNet50


import pfedplat.algorithm
from pfedplat.algorithm.Local.Local import Local
from pfedplat.algorithm.FedAvg.FedAvg import FedAvg
from pfedplat.algorithm.SplitGP.SplitGP import SplitGP
from pfedplat.algorithm.SplitGP.SplitGP_g import SplitGP_g
from pfedplat.algorithm.FedSGD.FedSGD import FedSGD
from pfedplat.algorithm.FedMGDA_plus.FedMGDA_plus import FedMGDA_plus
from pfedplat.algorithm.Ditto.Ditto import Ditto
from pfedplat.algorithm.FedProx.FedProx import FedProx
from pfedplat.algorithm.FedAMP.FedAMP import FedAMP
from pfedplat.algorithm.pFedMe.pFedMe import pFedMe
from pfedplat.algorithm.APFL.APFL import APFL
from pfedplat.algorithm.FedFomo.FedFomo import FedFomo
from pfedplat.algorithm.FedRep.FedRep import FedRep
from pfedplat.algorithm.FedROD.FedROD import FedROD
from pfedplat.algorithm.LFedAvg.LFedAvg import LFedAvg
from pfedplat.algorithm.FedPG.FedPG import FedPG
from pfedplat.algorithm.FedPG.FedPG_tau import FedPG_tau
from pfedplat.algorithm.FedPG.FedPG_d import FedPG_d
from pfedplat.algorithm.FedPG.FedPG_MOP import FedPG_MOP
from pfedplat.algorithm.FedPG.FedPG_nohistory import FedPG_nohistory
from pfedplat.algorithm.FedPG.FedPG_SGD import FedPG_SGD
from pfedplat.algorithm.FedPG.S_FedPG import S_FedPG


from pfedplat.dataloaders.separate_data import separate_data, create_data_pool
from pfedplat.dataloaders.DataLoader_cifar10_pat import DataLoader_cifar10_pat
from pfedplat.dataloaders.DataLoader_cifar10_dir import DataLoader_cifar10_dir
from pfedplat.dataloaders.DataLoader_fashion_pat import DataLoader_fashion_pat
from pfedplat.dataloaders.DataLoader_fashion_dir import DataLoader_fashion_dir
from pfedplat.dataloaders.DataLoader_cifar100_pat import DataLoader_cifar100_pat
from pfedplat.dataloaders.DataLoader_cifar100_dir import DataLoader_cifar100_dir


data_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/data/'
if not os.path.exists(data_folder_path):
    os.makedirs(data_folder_path)


pool_folder_path = os.path.dirname(os.path.abspath(__file__)) + '/pool/'
if not os.path.exists(pool_folder_path):
    os.makedirs(pool_folder_path)
