import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, DataLoader, InMemoryDataset
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T
import numpy as np
import os
from tqdm import tqdm, trange
from typing import Literal

from Encoding import ToPytorchData

def GetScales(dataset):
    scale_pos = 1.0/dataset.pos.abs().max()
    scale_x = 1/dataset.x.max(dim=0,keepdim=False)[0]
    return scale_pos,scale_x

class DEM_Dataset(InMemoryDataset):
    def __init__(self,file_name: str,data_split,
                 Dataset_type: Literal["train","validate","test"],
                 mode: Literal["cart","delta"],
                 force_reload=False,pre_transform=None, transform=None, pre_filter=None,
                 root: str = os.path.join(os.getcwd(),"Data")):
        
        self.raw_data_path = os.path.join(root,"raw")
        self.processed_data_path = os.path.join(root,"processed")
        self.file_name = file_name
        self.Dataset_type = Dataset_type
        self.mode = mode
        self.data_split = data_split
        super().__init__(root, transform, pre_transform,pre_filter,force_reload=force_reload)
        self.load(os.path.join(self.processed_data_path,self.processed_file_names[0]))

    @property 
    def raw_file_names(self):
        return[f"{self.file_name}_Data.npy",
               f"{self.file_name}_Topology.npy",
               f"{self.file_name}_BC.npy"]
    
    @property
    def processed_file_names(self):
        if self.pre_filter is None: return [f"{self.file_name}_{self.mode}_{self.Dataset_type}.pt"]
        else: return [f"{self.file_name}_{self.mode}_{self.Dataset_type}_init.pt"]
    
    def download(self):
        pass
    
    # Load data and split them according to dataset split
    def LoadSimTop(self,i):
        data = np.load(os.path.join(self.raw_data_path,self.raw_file_names[i]),allow_pickle=True)
        Daset_type_idx = {"train":0,"validate":1,"test":2}[self.Dataset_type]
        splits=np.array(self.data_split)*data.shape[0]
        return np.split(data,splits.astype(int))[Daset_type_idx]

    def process(self):
        data_list = []
        data_agr,top_agr,bc= [self.LoadSimTop(i) for i in [0,1,2]]
            
        if self.pre_filter is not None:
            simulations = self.pre_filter(simulations)

        for sim, top, bc in tqdm(zip(data_agr,top_agr,bc)):
            #R_avg = sim[0][:,3].mean()
            #topology = ConstructTopology(sim[0],bc,6*R_avg)-1
            for t in np.arange(len(sim)-1):
                par_data = sim[t]
                label_data = sim[t+1]
                topology = top[t]
                data = ToPytorchData(par_data,bc,None,topology,label_data)[0]
                data_list.append(data)

        if self.Dataset_type == "train":
            scale_pos, scale_x = GetScales(Batch.from_data_list(data_list))
            torch.save(scale_pos,os.path.join(self.processed_data_path,f"{self.file_name}_scale_pos.pt"))
            torch.save(scale_x,os.path.join(self.processed_data_path,f"{self.file_name}_scale_x.pt"))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]   
                
        self.save(data_list, os.path.join(self.processed_data_path,self.processed_file_names[0]))

from torch_geometric.transforms import BaseTransform
class NormalizePos(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self,file_name):
        self.path_scale_pos = os.path.join(os.getcwd(),"Data","processed",f"{file_name}_scale_pos.pt")
        self.path_scale_x   = os.path.join(os.getcwd(),"Data","processed",f"{file_name}_scale_x.pt")
        self.scale_pos = torch.load(self.path_scale_pos)
        self.scale_x   = torch.load(self.path_scale_x)
    def forward(self, data: Data) -> Data:
        data.pos *= self.scale_pos
        data.x   *= self.scale_x
        return data
    