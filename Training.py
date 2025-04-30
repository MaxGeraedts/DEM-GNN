import torch
import torch.nn.functional as F

from torch_geometric.data import Batch, Data, DataLoader, InMemoryDataset
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T

import numpy as np

from Encoding import AggregateRawData, save
from ML_functions import DEM_Dataset, NormalizePos, Trainer, GetModel

aggregate       = False
force_reload    = False
train           = True
dataset_name    = "N400_Mono"
model_ident     = "_model_1"

if aggregate == True:
    data_dir = "/home/20182319/Data"
    ArgsAggregation = AggregateRawData(data_dir,dataset_name)
    save(dataset_name,*ArgsAggregation)

data_split=[0.85, 0.95]
pre_transform = T.Compose([T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])
transform       = None

dataset_train     = DEM_Dataset(dataset_name, data_split,"train"   ,'delta', force_reload, pre_transform,transform)
dataset_val       = DEM_Dataset(dataset_name, data_split,"validate",'delta', force_reload, pre_transform,transform)
dataset_test      = DEM_Dataset(dataset_name, data_split,"test"    ,'delta', force_reload, pre_transform,transform)

if train == True:
    model = GetModel("N400",model_ident,
                    emb_dim=64,
                    edge_dim=4)
    
    trainer = Trainer(model, dataset_train,dataset_val,
                    batch_size=64,
                    lr=0.0000001,
                    epochs=1000,
                    model_name=f"{dataset_name}_{model_ident}")
    
    trainer.train_loop()