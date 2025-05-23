# Imports
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data, DataLoader, InMemoryDataset
from torch_geometric.nn import MessagePassing
import torch_geometric.transforms as T

import numpy as np
import os

from tqdm import tqdm, trange
from typing import Literal

from IPython.display import clear_output
import matplotlib.pyplot as plt
import json

from Encoding import ToPytorchData, ConstructTopology, TopologyFromPlausibleTopology

# Dataset
def GetScales(dataset):
    scale_pos = 1.0/dataset.pos.abs().max()
    scale_x = 1/dataset.x.max(dim=0,keepdim=False)[0]
    return scale_pos,scale_x

def DataMask(data,test_step: int = 20, val_step: int = 10):
    """Generates a boolean mask to split raw data into training, testing, and validation data

    Args:
        data (array): raw data array
        test_step (int, optional): Every nth steps which is included. Defaults to 20.
        val_step (int, optional): Every nth steps which are includeded. Defaults to 10.

    Returns:
        tuple: [train_mask, val_mask,test_mask]
    """
    test = np.zeros(data.shape[0]).astype(int)
    val = test.copy()
    test[0::test_step]=1
    val[1::val_step]=1
    train = test+val
    return np.invert(train.astype(bool)), val.astype(bool), test.astype(bool)

class DEM_Dataset(InMemoryDataset):
    def __init__(self,file_name: str,
                 Dataset_type: Literal["train","validate","test"],
                 mode: Literal["cart","delta"],
                 force_reload=False,pre_transform=None, transform=None, pre_filter=None,
                 root: str = os.path.join(os.getcwd(),"Data"),
                 super_tol: int = 6,
                 tol: float = 0,
                 noise_factor: float = 0):
        
        self.raw_data_path = os.path.join(root,"raw")
        self.processed_data_path = os.path.join(root,"processed")
        self.file_name = file_name
        self.Dataset_type = Dataset_type
        self.mode = mode
        self.super_tol = super_tol
        self.tol = tol
        self.noise_factor = noise_factor
        super().__init__(root, transform, pre_transform,pre_filter,force_reload=force_reload)
        self.load(os.path.join(self.processed_data_path,self.processed_file_names[0]))

    @property 
    def raw_file_names(self):
        return[f"{self.file_name}_Data.npy",
               f"{self.file_name}_Topology.npy",
               f"{self.file_name}_BC.npy"]
    
    @property
    def processed_file_names(self):
        return [f"{self.file_name}_{self.Dataset_type}.pt"]
    
    def download(self):
        pass
    
    # Load data and split them according to dataset split
    def LoadSimTop(self,i):
        data = np.load(os.path.join(self.raw_data_path,self.raw_file_names[i]),allow_pickle=True)
        type_idx = {"train":0,"validate":1,"test":2}[self.Dataset_type]
        mask =  DataMask(data)[type_idx]
        return data[mask]

    def process(self):
        data_list = []
        data_agr,top_agr,bc= [self.LoadSimTop(i) for i in [0,1,2]]
            
        if self.pre_filter is not None:
            simulations = self.pre_filter(simulations)

        print(f"Collecting {self.Dataset_type} data")
        for sim, top, bc in tqdm(zip(data_agr,top_agr,bc)):
            R_avg = sim[0][:,3].mean()
            super_topology = ConstructTopology(sim[0],bc,self.super_tol)-1
            for t in np.arange(len(sim)-1):
                par_data = sim[t].copy()
                standard_deviation = self.noise_factor*R_avg
                noise = np.array(standard_deviation*torch.randn((par_data.shape[0],3)))
                par_data[:,:3]+=noise

                label_data = sim[t+1]
                BC_t = bc.copy()
                BC_t[:,:3] = bc[:,:3]+(t+1)*bc[:,-3:]
                topology = TopologyFromPlausibleTopology(super_topology,par_data,BC_t,self.tol)

                data = ToPytorchData(par_data,BC_t,0,topology,label_data,center=False)[0]
                data_list.append(data)

        if self.Dataset_type == "train":
            scale_pos, scale_x = GetScales(Batch.from_data_list(data_list))
            torch.save(scale_pos,os.path.join(self.processed_data_path,f"{self.file_name}_scale_pos.pt"))
            torch.save(scale_x,os.path.join(self.processed_data_path,f"{self.file_name}_scale_x.pt"))

        print(f"Pre-processing {self.Dataset_type} data")
        if self.pre_transform is not None:
            #self.pre_transform = T.Compose([NormalizePos(self.file_name),self.pre_transform])
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

# Model
class RelPosConv(MessagePassing):
    def __init__(self, emb_dim, msg_dim, out_channels, aggr = 'mean'):
        super().__init__(aggr=aggr)
        self.edge_mlp = torch.nn.Linear(emb_dim+emb_dim,msg_dim)
        self.update_mlp = torch.nn.Linear(emb_dim+msg_dim,out_channels)
        self.reset_parameters()

    def forward(self,x,edge_attr,edge_index):
        out = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr],dim=1)
        return self.edge_mlp(tmp)
    
    def update(self,aggr_out,x):
        cat = torch.cat([x, aggr_out],dim=1)
        return self.update_mlp(cat)
    
class GCONV_Model_RelPos(torch.nn.Module):
    def __init__(self,msg_num=3, emb_dim=64, msg_dim=64, node_dim=7, edge_dim=4, out_dim = 3):
        super(GCONV_Model_RelPos,self).__init__()
        self.msg_num = msg_num
        self.emb_dim = emb_dim
        self.msg_dim = msg_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.node_embed = torch.nn.Linear(node_dim,emb_dim)
        self.edge_embed = torch.nn.Linear(edge_dim,emb_dim)
        self.conv = [None]* msg_num
        for k in range(msg_num):
            self.conv[k] = RelPosConv(emb_dim,msg_dim,emb_dim)
            self.conv[k].double().to(torch.device('cuda' if torch.cuda.is_available()else 'cpu'))
        #self.conv2 = RelPosConv(emb_dim,msg_dim,emb_dim)
        #self.conv3 = RelPosConv(emb_dim,msg_dim,emb_dim)
        #self.conv4 = RelPosConv(emb_dim,msg_dim,emb_dim)
        #self.conv5 = RelPosConv(emb_dim,msg_dim,emb_dim)
        #self.conv6 = RelPosConv(emb_dim,msg_dim,emb_dim)
        self.decoder = torch.nn.Linear(emb_dim,out_dim)
        self.double()
        
    def forward(self,data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.node_embed(x)
        x = F.relu(x)
        edge_attr = self.edge_embed(edge_attr)
        edge_attr = F.relu(edge_attr)
        for k in range(self.msg_num):
            x = self.conv[k](x, edge_attr, edge_index)
            x = F.relu(x)
        #x = self.conv2(x, edge_attr, edge_index)
        #x = F.relu(x)
        #x = self.conv3(x, edge_attr, edge_index)
        #x = F.relu(x)
        #x = self.conv4(x, edge_attr, edge_index)
        #x = F.relu(x)
        #x = self.conv5(x, edge_attr, edge_index)
        #x = F.relu(x)
        #x = self.conv6(x, edge_attr, edge_index)
        #x = F.relu(x)
        x= self.decoder(x)
        return x
    
# Training
class Trainer:
    def __init__(self,model,dataset_train,dataset_val,batch_size,lr,epochs,model_name,loss_fn=torch.nn.MSELoss()):
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.model_name = model_name

        self.device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
        print("Device: ", self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.train_dl = self.make_data_loader(dataset_train, shuffle=True)
        self.val_dl = self.make_data_loader(dataset_val, shuffle=False)

    def make_data_loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def loss_batch(self, batch,opt=None):
        out = self.model(batch)
        mask = np.concatenate(batch.mask)
        loss =self.loss_fn(out[mask], batch.y)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item()

    def batch_loop(self, dataloader, loss_list, opt=None):
        mean_loss = 0
        for i, batch in enumerate(dataloader):
            batch_loss = self.loss_batch(batch.to(self.device), opt)
            mean_loss += batch_loss
        mean_loss /= i
        loss_list.append(mean_loss)
        return mean_loss,loss_list

    def train_loop(self):
        train_loss, val_loss = [], []
        best_model_loss = np.inf
        for epoch in tqdm(range(self.epochs)):
            self.model.train()  
            mean_train_loss, train_loss = self.batch_loop(self.train_dl,train_loss,self.optimizer)

            self.model.eval()
            with torch.inference_mode():
                mean_val_loss, val_loss = self.batch_loop(self.val_dl,val_loss)

            if mean_val_loss < best_model_loss:
                best_model_loss = mean_val_loss
                torch.save(self.model.state_dict(),os.path.join(os.getcwd(),"Models",self.model_name))
            
            if epoch % 100 == 0:
                np.save(os.path.join(os.getcwd(),"Models",f"{self.model_name}_Training_Loss"),train_loss)
                np.save(os.path.join(os.getcwd(),"Models",f"{self.model_name}_Validation_Loss"),val_loss)

            print(f"\nEpoch: {epoch:03d}  |  Mean Train Loss: {mean_train_loss:.10f}  |  Mean Validation Loss: {mean_val_loss:.10f}",flush=True)


def GetModel(dataset_name,model_ident,msg_num=3,emb_dim=64,msg_dim=64,edge_dim=4):
    try: 
        model_name = os.path.join(os.getcwd(),"Models",f"{dataset_name}_{model_ident}")
        with open(f"{model_name}.json") as json_file: 
            settings = json.load(json_file)
        model = GCONV_Model_RelPos(msg_num=settings["msg_num"],
                                   emb_dim=settings["emb_dim"],
                                   msg_dim=settings["msg_dim"],
                                   node_dim=settings["node_dim"],
                                   edge_dim=settings["edge_dim"])
        model.load_state_dict(torch.load(model_name))
        print("Loaded model")
    except: 
        print("No Trained model")
        model = GCONV_Model_RelPos(msg_num=msg_num,
                                   emb_dim=emb_dim,
                                   msg_dim=msg_dim,
                                   edge_dim=edge_dim)
    return model

def SaveModelInfo(model,dataset_name,model_ident):
    ModelInfo = {"msg_num":model.msg_num,
                 "emb_dim":model.emb_dim,
                 "msg_dim":model.msg_dim,
                 "node_dim":model.node_dim,
                 "edge_dim":model.edge_dim,}
    filename = os.path.join(os.getcwd(),"Models",f"{dataset_name}_{model_ident}.json")
    with open(filename,'w') as f: 
        json.dump(ModelInfo,f)

