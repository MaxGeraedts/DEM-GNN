# Imports
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, EdgeConv, GCNConv
from torch_geometric.nn.models import MLP
import torch_geometric.transforms as T

import numpy as np
import numpy.typing as npt
import os

from tqdm import tqdm, trange
from typing import Literal

import json

from Encoding import ToPytorchData, ConstructTopology, TopologyFromPlausibleTopology, GetLength, load

# Rollout
def MaskTestData(dataset_name:str,dataset_type: Literal["train","validate","test"])->tuple[npt.NDArray,npt.NDArray,npt.NDArray]:
    """Masks out data from rawdata arrays

    Returns:
        tuple: [data, topology, boundary_conditions]
    """
    type_dic= {"train": 0, "validate":1, "test":2}
    loaded_data = load(dataset_name)
    mask = DataMask(loaded_data[0],)[type_dic[dataset_type]]
    test_data = [data[mask] for data in loaded_data]
    return test_data
    
class LearnedSimulator:
    def __init__(self,model,scale_function,super_tol:int = 6,tol:int = 0, transform=None,timesteps:int=100):
        self.device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
        self.model = model.to(self.device)
        self.rescale = scale_function
        self.timesteps = timesteps
        self.super_tol = super_tol
        self.tol = tol
        self.transform = transform
        self.bundle_size = model.bundle_size

    def BCrollout(self,show_tqdm:bool)->list[npt.NDArray]:
        if show_tqdm: print("Calculating BC")
        BC_rollout = np.empty((GetLength(self.par_data),6,9))
        BC_t=np.copy(self.BC)
        for t in trange(GetLength(self.par_data),disable = not show_tqdm):
            BC_t[:,:3] = self.BC[:,:3]+(t+1)*self.BC[:,-3:] 
            BC_rollout[t] = BC_t
        return BC_rollout
    
    def GroundTruth_Rollout(self,show_tqdm:bool)->npt.NDArray:
        if show_tqdm: print("Collecting Ground Truth Rollout")
        GroundTruth = np.empty(GetLength(self.par_data),dtype=object)
        for t in trange(GetLength(self.par_data),disable = not show_tqdm):
            MatlabTopology = TopologyFromPlausibleTopology(self.super_topology,self.par_data[t],self.BC_rollout[t],self.tol)
            GroundTruth[t] = ToPytorchData(self.par_data[t],self.BC_rollout[t],self.tol,MatlabTopology)[0]
            GroundTruth[t].MatlabTopology = MatlabTopology
        return GroundTruth
    
    def Rollout_Step(self,par_inp:npt.NDArray, BC:npt.NDArray,MatlabTopology:npt.NDArray, ML_Rollout: list[object] = None):

        # Convert raw data to PyTorch Graphdata
        input_data = ToPytorchData(par_inp,BC,self.tol,MatlabTopology)[0]
        if self.transform is not None: input_data = self.transform(input_data)
        input_data.to(self.device)

        # Run ML Model
        output = self.model(input_data)
        output = self.rescale(output,self.device).cpu()
        output = np.stack(np.split(output,output.shape[1]/3,axis=1))
        # With displacement vectors update particle positions and topology
        for displacement in output:
            par_inp[:,:3] = par_inp[:,:3]+displacement[input_data.mask]
            MatlabTopology = TopologyFromPlausibleTopology(self.super_topology,par_inp,BC,self.tol)

            if ML_Rollout is not None:
                data = ToPytorchData(par_inp,BC,self.tol,MatlabTopology)[0]
                data.MatlabTopology = MatlabTopology
                ML_Rollout.append(data)

            BC[:,:3] += BC[:,-3:]

        return par_inp, BC, MatlabTopology
    
    def MLRollout(self,show_tqdm:bool)->list[object]:
        if show_tqdm: print("Calculating Learned Rollout")
        with torch.inference_mode():
            # Initiate Rollout
            ML_Rollout = []
            par_inp = self.par_data[0]
            BC = self.BC
            BC[:,:3] += BC[:,-3:]
            MatlabTopology = TopologyFromPlausibleTopology(self.super_topology,par_inp,BC,self.tol)
            data = ToPytorchData(par_inp,BC,self.tol,MatlabTopology)[0]
            data.MatlabTopology = MatlabTopology
            ML_Rollout.append(data)

            # Rollout
            BC[:,:3] += BC[:,-3:]
            for t in tqdm(range(1,self.timesteps,self.bundle_size),initial=1,disable = not show_tqdm):
                par_inp, BC, MatlabTopology = self.Rollout_Step(par_inp, BC, MatlabTopology,ML_Rollout)

        return ML_Rollout

    def Rollout(self,data_agr: np.ndarray, top_agr: np.ndarray,BC_agr:np.ndarray, i:int,show_tqdm: bool = False):
        self.BC = BC_agr[i].copy()
        self.par_data = data_agr[i].copy()
        self.topology = top_agr[i].copy()
        self.BC_rollout = self.BCrollout(show_tqdm)
        self.super_topology = ConstructTopology(self.par_data[0],self.BC_rollout[0],self.super_tol)-1
        self.GroundTruth = self.GroundTruth_Rollout(show_tqdm)
        self.ML_rollout = self.MLRollout(show_tqdm)

# Dataset
def GetScales(dataset,dataset_name):
    scales = {"scale_x":    dataset.x.max(dim=0,keepdim=False)[0].tolist(),
              "edge_mean":  dataset.edge_attr.mean(dim=0).tolist(),
              "edge_std":   dataset.edge_attr.std(dim=0).tolist(),
              "y_mean":     dataset.y.mean(dim=0).tolist(),
              "y_std":      dataset.y.std(dim=0).tolist()}
    
    filename = os.path.join(os.getcwd(),"Data","processed",f"{dataset_name}_scales.json")
    with open(filename,'w') as f: 
        json.dump(scales,f)
    return scales

class Rescale:
    """Rescale output based on standardization during training
    """
    def __init__(self,dataset_name):
        self.filename = os.path.join(os.getcwd(),"Data","processed",f"{dataset_name}_scales")
        with open(f"{self.filename}.json") as json_file: 
            self.scales = json.load(json_file)
        self.y_mean = self.scales["y_mean"]
        self.y_std = self.scales["y_std"]

    def __call__(self, output,device):
        """Rescale output based on training standardization statistics

        Args:
            output (Tensor): Model output tensor

        Returns:
            Tensor: Rescaled model output tensot
        """
        output *= torch.tensor(self.y_std).to(device)
        output += torch.tensor(self.y_mean).to(device)
        return output
    

class NormalizeData(T.BaseTransform):
    r"""Scales node features to :math:`(0, 1)`. Standardizes edge attributes and optionally labels (zero mean, unit variance)
    """
    def __init__(self,dataset_name):
        filename = os.path.join(os.getcwd(),"Data","processed",f"{dataset_name}_scales")
        with open(f"{filename}.json") as json_file: 
            self.scales = json.load(json_file)

    def forward(self, data: Data) -> Data:
        data.x /= torch.tensor(self.scales["scale_x"])

        data.edge_attr -= torch.tensor(self.scales["edge_mean"])
        data.edge_attr /= torch.tensor(self.scales["edge_std"])
        
        if data.y is not None:
            data.y -= torch.tensor(self.scales["y_mean"])
            data.y /= torch.tensor(self.scales["y_std"])

        return data
    
class NormalizePos(T.BaseTransform):
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
                 force_reload=False,pre_transform=None, transform=None, pre_filter=None,
                 root: str = os.path.join(os.getcwd(),"Data"),
                 super_tol: int = 6,
                 tol: float = 0,
                 noise_factor: float = 0,
                 push_forward_step_max: int = 0,
                 bundle_size: int = 1,
                 model = None):
        
        self.raw_data_path = os.path.join(root,"raw")
        self.processed_data_path = os.path.join(root,"processed")
        self.file_name = file_name
        self.Dataset_type = Dataset_type
        self.super_tol = super_tol
        self.tol = tol
        self.noise_factor = noise_factor
        self.forward_step_max = push_forward_step_max
        self.bundle_size = bundle_size
        self.model = model

        super().__init__(root, transform, pre_transform,pre_filter,force_reload=force_reload)
        self.load(os.path.join(self.processed_data_path,self.processed_file_names[0]))

    @property 
    def raw_file_names(self):
        return[f"{self.file_name}_Data.npy",
               f"{self.file_name}_Topology.npy",
               f"{self.file_name}_BC.npy"]
    
    @property
    def processed_file_names(self):
        processed_file_name = f"{self.file_name}_bund{self.bundle_size}_{self.Dataset_type}"
        if self.forward_step_max > 0: 
            processed_file_name += f"_push{self.forward_step_max}"
        processed_file_name  += ".pt"  
        return [processed_file_name]
    
    def download(self):
        pass
    
    # Load data and split them according to dataset split
    def LoadSimTop(self,i):
        data = np.load(os.path.join(self.raw_data_path,self.raw_file_names[i]),allow_pickle=True)
        type_idx = {"train":0,"validate":1,"test":2}[self.Dataset_type]
        mask =  DataMask(data)[type_idx]
        return data[mask]
    
    def SliceAndReshapeData(self,par_data,t,push_forward_steps):
        pos_slice = par_data[t+push_forward_steps*self.bundle_size:t+(push_forward_steps+1)*self.bundle_size,:,:3]
        #pos_slice_2D = np.reshape(np.swapaxes(pos_slice,0,1),(-1,3*self.bundle_size))
        pos_slice_2D = np.concatenate([pos for pos in pos_slice],axis=1)
        return pos_slice_2D
    
    def process(self):
        data_list = []
        data_agr,top_agr,bc= [self.LoadSimTop(i) for i in [0,1,2]]
            
        if self.pre_filter is not None:
            simulations = self.pre_filter(simulations)

        if self.forward_step_max > 0:
            Simulation = LearnedSimulator(self.model,
                                          scale_function=Rescale(f"{self.file_name}_bund{self.bundle_size}"),
                                          transform = T.Compose([self.pre_transform,NormalizeData(f"{self.file_name}_bund{self.bundle_size}")]))
            self.Rollout_step = Simulation.Rollout_Step

        print(f"Collecting {self.Dataset_type} data")
        for sim, top, bc in tqdm(zip(data_agr,top_agr,bc),total=bc.shape[0]):
            R_avg = sim[0][:,3].mean()
            self.super_topology = ConstructTopology(sim[0],bc,self.super_tol)-1
            if self.forward_step_max > 0: Simulation.super_topology = self.super_topology
            for t in range(len(sim)-1*self.bundle_size*(self.forward_step_max+1)):
                par_inp = sim[t].copy()
                BC = bc.copy()
                BC[:,:3] = bc[:,:3]+(t+1)*bc[:,-3:]

                push_forward_steps = np.random.randint(0,self.forward_step_max+1)
                MatlabTopology = TopologyFromPlausibleTopology(self.super_topology,par_inp,BC,self.tol)
                with torch.inference_mode():
                    for forward_step in range(push_forward_steps):
                        par_inp, BC, MatlabTopology = self.Rollout_step(par_inp, BC, MatlabTopology)

                if self.noise_factor > 0:
                    standard_deviation = self.noise_factor*R_avg
                    noise = np.array(standard_deviation*torch.randn((par_inp.shape[0],3)))
                    par_inp[:,:3]+=noise

                BC[:,:3] += BC[:,-3:]
                pos_slice        = self.SliceAndReshapeData(sim,t  ,push_forward_steps)
                pos_target_slice = self.SliceAndReshapeData(sim,t+1,push_forward_steps)
                displacements = pos_target_slice-pos_slice

                data = ToPytorchData(par_inp,BC,0,MatlabTopology,label_data=displacements)[0]
                data.push_forward_steps = push_forward_steps
                data_list.append(data)

        print(f"Pre-processing {self.Dataset_type} data")
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        print(f"Normalizing {self.Dataset_type} data")    
        if self.Dataset_type == "train" and self.forward_step_max == 0:
            GetScales(Batch.from_data_list(data_list),f"{self.file_name}_bund{self.bundle_size}")
        self.normalize = NormalizeData(f"{self.file_name}_bund{self.bundle_size}")
        data_list = [self.normalize(data) for data in tqdm(data_list)]
                
        self.save(data_list, os.path.join(self.processed_data_path,self.processed_file_names[0]))

# Model
class RelPosConv(MessagePassing):
    def __init__(self, emb_dim, hidden_dim, out_channels,num_layers, aggr = 'mean'):
        super().__init__(aggr=aggr)
        self.msg_mlp = MLP(in_channels=2*emb_dim,hidden_channels=hidden_dim,out_channels=out_channels,num_layers=num_layers)
        self.update_mlp = MLP(in_channels=2*emb_dim,hidden_channels=hidden_dim,out_channels=out_channels,num_layers=num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.msg_mlp.reset_parameters()
        self.update_mlp.reset_parameters()

    def forward(self,x,edge_attr,edge_index):
        out = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        return out
    
    def message(self, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr],dim=1)
        return self.msg_mlp(tmp)
    
    def update(self,aggr_out,x):
        cat = torch.cat([x, aggr_out],dim=1)
        return self.update_mlp(cat)
    
class GCONV_Model_RelPos(torch.nn.Module):
    def __init__(self,msg_num:int=3, emb_dim:int=64, hidden_dim:int=64, node_dim:int=7, edge_dim:int=4, bundle_size:int = 1,num_layers:int = 2):
        super(GCONV_Model_RelPos,self).__init__()
        self.msg_num = msg_num
        self.emb_dim = emb_dim
        self.hidden_dim = emb_dim
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.bundle_size = bundle_size
        self.out_dim = 3*bundle_size
        self.num_layers = num_layers
        self.node_embed = MLP(in_channels=node_dim,hidden_channels=hidden_dim,out_channels=emb_dim,num_layers=num_layers,norm=None)
        self.edge_embed = MLP(in_channels=edge_dim,hidden_channels=hidden_dim,out_channels=emb_dim,num_layers=num_layers,norm=None)
        self.convs = torch.nn.ModuleList()
        for k in range(msg_num):
            self.convs.append(RelPosConv(emb_dim=emb_dim,hidden_dim=hidden_dim,out_channels=emb_dim,num_layers=num_layers))
        self.decoder = MLP(in_channels=emb_dim,hidden_channels=hidden_dim,out_channels=self.out_dim,num_layers=num_layers,norm=None)
        self.double()
        
    def forward(self,data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        for conv in self.convs:
            x = conv(x, edge_attr, edge_index)
        x = self.decoder(x)
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


def GetModel(model_name,msg_num=3,emb_dim=64,node_dim=7,edge_dim=4,num_layers=2,bundle_size=1):
    try: 
        model_path = os.path.join(os.getcwd(),"Models",f"{model_name}")
        if model_name[-5:] == '_Push': 
            with open(f"{model_path[:-5]}_ModelInfo.json") as json_file: settings = json.load(json_file)
        else:
            with open(f"{model_path}_ModelInfo.json") as json_file: settings = json.load(json_file)

        model = GCONV_Model_RelPos(msg_num=settings["msg_num"],
                                   emb_dim=settings["emb_dim"],
                                   hidden_dim=settings["hidden_dim"],
                                   node_dim=settings["node_dim"],
                                   edge_dim=settings["edge_dim"],
                                   bundle_size=settings["bundle_size"],
                                   num_layers=settings["num_layers"])
        model.load_state_dict(torch.load(model_path))
        msg = "Loaded model"
        print(f"{msg} {model_name}")
    except: 
        msg = "No Trained model"
        print(msg)
        model = GCONV_Model_RelPos(msg_num,
                                   emb_dim=emb_dim,
                                   hidden_dim=emb_dim,
                                   node_dim=node_dim,
                                   edge_dim=edge_dim,
                                   bundle_size=bundle_size,
                                   num_layers=num_layers)
    return model, msg

def SaveModelInfo(model,dataset_name:str,model_ident:str):
    """Saves Model parameters to a dictionary JSON file

    Args:
        model (Object): Pytorch model as defined by RelPosConv class
        dataset_name (string): Name of the dataset
        model_ident (string): Identifier for the model
    """
    ModelInfo = {"msg_num":model.msg_num,
                 "emb_dim":model.emb_dim,
                 "hidden_dim":model.hidden_dim,
                 "node_dim":model.node_dim,
                 "edge_dim":model.edge_dim,
                 "out_dim": model.out_dim,
                 "num_layers":model.num_layers,
                 "bundle_size":model.bundle_size}
    filename = os.path.join(os.getcwd(),"Models",f"{dataset_name}_{model_ident}_ModelInfo.json")
    with open(filename,'w') as f: 
        json.dump(ModelInfo,f)

def SaveTrainingInfo(dataset,trainer):
    TrainingInfo = {"super_tol":dataset.super_tol,
                    "tol":dataset.tol,
                    "noise_factor":dataset.noise_factor,
                    "batch_size":trainer.batch_size,
                    "learning_rate":trainer.lr,
                    "bundle_size":dataset.bundle_size,
                    "push_forward_step_max":dataset.forward_step_max}
    
    if dataset.model is not None: 
        TrainingInfo["model_name"] = trainer.model_name

    filename = os.path.join(os.getcwd(),"Models",f"{trainer.model_name}_TrainingInfo.json")
    with open(filename,'w') as f: 
        json.dump(TrainingInfo,f)