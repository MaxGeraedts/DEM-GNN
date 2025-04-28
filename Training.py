import torch
import torch.nn.functional as F

from torch_geometric.data import Batch, Data, DataLoader, InMemoryDataset
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T

import numpy as np

from Encoding import AggregateRawData, save
from ML_functions import DEM_Dataset, NormalizePos

aggregate = True

if aggregate == True:
    dataset_name = "Scalability_Mono"
    data_dir = "/home/20182319/Data/Scalability_Mono"
    ArgsAggregation = AggregateRawData(data_dir,dataset_name)
    save(dataset_name,*ArgsAggregation)

data_split=[0.85, 0.95]
pre_transform = T.Compose([NormalizePos(dataset_name),
                           T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])
transform       = None
force_reload    = True

dataset_train     = DEM_Dataset(dataset_name, data_split,"train"   ,'delta', force_reload, pre_transform,transform)
dataset_val       = DEM_Dataset(dataset_name, data_split,"validate",'delta', force_reload, pre_transform,transform)
dataset_test      = DEM_Dataset(dataset_name, data_split,"test"    ,'delta', force_reload, pre_transform,transform)