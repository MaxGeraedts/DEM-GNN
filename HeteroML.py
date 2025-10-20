import warnings
import os
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Literal

import torch
import numpy as np
from torch import Tensor

from torch_geometric.nn.models import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.hetero import check_add_self_loops
from torch_geometric.data import Batch, Data,HeteroData, InMemoryDataset
import torch_geometric.transforms as T

from Encoding import ToHeteroData,ConstructTopology,TopologyFromPlausibleTopology, ToPytorchData
from ML_functions import LearnedSimulator,Trainer, NormalizeData,Rescale, DataMask, GetScales

def MakeDIRs(dataset_name):
    root = os.getcwd()
    model_dir = os.path.join(root,"Models",dataset_name)
    data_raw_dir = os.path.join(root,"Data","raw",dataset_name)
    data_proc_dir = os.path.join(root,"Data","processed",dataset_name)

    for dir in [model_dir,data_proc_dir]:
        if not os.path.isdir(dir): os.mkdir(dir)

class LearnedSimulatorHetero(LearnedSimulator):
    def __init__(self, model, scale_function, super_tol = 6, tol = 0, transform=None, timesteps = 100, device = 'cuda'):
        super().__init__(model, scale_function, super_tol, tol, transform, timesteps, device)

    def Rollout_Step(self, par_inp, BC, MatlabTopology, ML_Rollout = None):
        # Convert raw data to PyTorch Graphdata
        input_data = ToHeteroData(par_inp,MatlabTopology,BC)
        if self.transform is not None: input_data = self.transform(input_data)
        input_data.to(self.device)

        # Run ML Model
        with torch.inference_mode():
            output = self.model(input_data)[0]
        output = self.rescale(output).cpu()
        output = np.stack(np.split(output,output.shape[1]/3,axis=1))
        # With displacement vectors update particle positions and topology
        for displacement in output:
            par_inp[:,:3] = par_inp[:,:3]+displacement
            MatlabTopology = TopologyFromPlausibleTopology(self.super_topology,par_inp,BC,self.tol)

            if ML_Rollout is not None:
                data,MatlabTopology = ToPytorchData(par_inp,BC,self.tol,MatlabTopology)[:2]
                data.MatlabTopology = MatlabTopology
                ML_Rollout.append(data)

            BC[0] += BC[1]

        return par_inp, BC, MatlabTopology

class EdgeConv(MessagePassing):
    def __init__(self,emb_dim,hidden_dim,num_layers, aggr = 'mean', ):
        super().__init__(aggr=aggr)
        self.msg_mlp = MLP(in_channels=8+2*emb_dim,hidden_channels=hidden_dim,out_channels=hidden_dim,num_layers=num_layers)
        self.node_mlp = MLP(in_channels=hidden_dim+emb_dim,hidden_channels=hidden_dim,out_channels=emb_dim,num_layers=num_layers)
        self.edge_mlp = MLP(in_channels=hidden_dim,hidden_channels=hidden_dim,out_channels=emb_dim,num_layers=num_layers)


    def forward(self,x,edge_attr,edge_index):
        node_emb = self.propagate(edge_index,x=x,edge_attr=edge_attr)
        edge_emb = self.edge_updater(edge_index,x=x,edge_attr=edge_attr)
        return node_emb, edge_emb
        
    def message(self, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr],dim=1)
        return self.msg_mlp(tmp)
    
    def edge_update(self, x_j, edge_attr):
        message = self.message(x_j,edge_attr)
        return self.edge_mlp(message)

class HeteroDEMGNN(torch.nn.Module):
    def __init__(self,dataset_name,metadata, msg_num,emb_dim,hidden_dim,num_layers):
        super().__init__()
        self.dataset_name = dataset_name
        self.scale_name = f"{dataset_name}_Hetero"
        self.nodetypes = metadata[0]
        self.edgetypes = metadata[1]

        self.msg_num = msg_num
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.edge_embed_mlp = MLP(in_channels=4,hidden_channels=emb_dim,out_channels=emb_dim,num_layers=num_layers)
        self.node_embed_mlp = torch.nn.ModuleDict()
        for nodetype in metadata[0]:
            embed_mlp =MLP(in_channels=3,hidden_channels=emb_dim,out_channels=emb_dim,num_layers=num_layers)
            self.node_embed_mlp[nodetype] = embed_mlp

        self.convs = torch.nn.ModuleList()
        self.node_updaters = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        for _ in range(msg_num):
            conv = HeteroConvEdge({edgetype:EdgeConv(emb_dim,hidden_dim,num_layers)
                                   for edgetype in self.edgetypes},aggr="mean")
            decoder = MLP(in_channels=emb_dim,hidden_channels=hidden_dim,out_channels=3,num_layers=num_layers)
            node_updater = MLP(in_channels=emb_dim+hidden_dim,hidden_channels=hidden_dim,out_channels=emb_dim,num_layers=num_layers)
            self.convs.append(conv)
            self.node_updaters.append(node_updater)
            self.decoders.append(decoder)
        self.double()

    def EmbedNodes(self,x_dict:dict) -> dict:
        node_emb_dict = {}
        for nodetype in self.nodetypes:
            node_emb_dict[nodetype] = self.node_embed_mlp[nodetype](x_dict[nodetype])
        return node_emb_dict
    
    def EmbedEdges(self,edge_attr_dict):
        edge_emb_dict = {}
        for edgetype in self.edgetypes:
            edge_emb_dict[edgetype] = self.edge_embed_mlp(edge_attr_dict[edgetype])
        return edge_emb_dict
    
    def MergeEdgeDicts(self,edge_ref_dict,edge_attr_dict,edge_emb_dict):
        edge_inp_dict = {}
        for edgetype in self.edgetypes:
            tensors = (edge_ref_dict[edgetype],edge_attr_dict[edgetype],edge_emb_dict[edgetype])
            edge_inp_dict[edgetype] = torch.cat(tensors,dim=1)
        return edge_inp_dict

    def UpdateNodeEmbeddings(self,x_dict,node_aggr_dict,node_updater):
        for nodetype in self.nodetypes:
            tmp = torch.cat([x_dict[nodetype],node_aggr_dict[nodetype]],dim=1)
            x_dict[nodetype] = node_updater(tmp)
        return x_dict

    def UpdateWallpointPos(self,data,displacement,normal):
        particle_idx = data.edge_index_dict[self.edgetypes[1]][0]
        edge_disp = displacement[particle_idx]
        delta_wallpoint = edge_disp-(edge_disp*normal)*normal
        data.pos_dict['wallpoint']+=delta_wallpoint
        return data

    def UpdateGeometry(self,data, displacement, normal):
        transform = T.Compose([CartesianHetero(cat= False),
                               DistanceHetero(cat = True),
                               NormalizeHeteroData(self.dataset_name,self.scale_name,edge_only=True)])
        rescale_output = Rescale(self.dataset_name,self.scale_name)

        displacement = rescale_output(displacement)
        data.pos_dict['particle']+=displacement
        #data = self.UpdateWallpointPos(data,displacement,normal)
        data = transform(data)

        return data

    def forward(self,data):
        normal = data.x_dict['wallpoint'].detach().clone()
        data.edge_ref_dict = data.edge_attr_dict.copy()
        data.x_dict = self.EmbedNodes(data.x_dict)
        data.edge_emb_dict = self.EmbedEdges(data.edge_attr_dict)

        displacement = 0
        disp_list = []
        for conv,node_updater,decoder in zip(self.convs,self.node_updaters,self.decoders):
            data.edge_inp_dict = self.MergeEdgeDicts(data.edge_ref_dict,data.edge_attr_dict,data.edge_emb_dict)
            data.node_aggr_dict, data.edge_emb_dict = conv(data.x_dict,data.edge_inp_dict,data.edge_index_dict)
            data.x_dict = self.UpdateNodeEmbeddings(data.x_dict,data.node_aggr_dict,node_updater)
            displacement_k = decoder(data.x_dict['particle'])
            data = self.UpdateGeometry(data,displacement_k,normal)
            displacement += displacement_k
            disp_list.append(displacement_k)
        return displacement, disp_list

class HeteroTrainer(Trainer):
    def __init__(self, model, batch_size, lr, epochs, dataset_name, model_ident):
        super().__init__(model, batch_size, lr, epochs, dataset_name, model_ident)

    def loss_batch(self, batch, opt=None):
        batch.device = self.device
        out = self.model(batch)[0]
        #print(out)
        #print(batch['particle'].y)
        loss =self.loss_fn(out, batch['particle'].y)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item()
    
class CartesianHetero(T.BaseTransform):
    def __init__(self,cat:bool=True):
        self.cat=cat

    def AddFeatureToEdge(self,data,origin:str,edge_type:str,destination:str):
        try:
            temp = data[edge_type].edge_attr
        except:
            temp = None
            
        (or_idx,dest_idx) = data[edge_type].edge_index
        cart = data[origin].pos[or_idx]-data[destination].pos[dest_idx]

        if temp is not None and self.cat:
            data[edge_type].edge_attr = torch.cat([temp, cart.type_as(temp)], dim=-1)
        else:
            data[edge_type].edge_attr = cart
        return data

    def forward(self, data:HeteroData) -> HeteroData:
        for edgetype in data.metadata()[1]:
            data = self.AddFeatureToEdge(data,*edgetype)
        return data

class DistanceHetero(CartesianHetero):
    def __init__(self, cat:bool = True):
        super().__init__(cat)
    def AddFeatureToEdge(self, data, origin:str, edge_type:str, destination:str):
        try:
            temp = data[edge_type].edge_attr
        except:
            temp = None
            
        (or_idx,dest_idx) = data[edge_type].edge_index
        cart = data[origin].pos[or_idx]-data[destination].pos[dest_idx]
        dist = torch.linalg.vector_norm(cart,dim=1,keepdim=True)

        if temp is not None and self.cat:
            data[edge_type].edge_attr = torch.cat([temp, dist.type_as(temp)], dim=-1)
        else:
            data[edge_type].edge_attr = dist
        return data

class NormalizeHeteroData(NormalizeData):
    def __init__(self, dataset_name:str, scale_name:str, edge_only:bool):
        self.edge_only = edge_only
        super().__init__(dataset_name, scale_name) 
    
    def forward(self, data: HeteroData) -> HeteroData:
        device = data['particle'].x.device
        if self.edge_only == False:
            data['particle'].x /= torch.tensor(self.scales["scale_x"]).to(device)

        for edgetype in data.metadata()[1]:
            data[edgetype].edge_attr -= torch.tensor(self.scales["edge_mean"]).to(device)
            data[edgetype].edge_attr /= torch.tensor(self.scales["edge_std"]).to(device)

        if hasattr(data['particle'],'y') and self.edge_only == False:
            data['particle'].y -= torch.tensor(self.scales["y_mean"]).to(device)
            data['particle'].y /= torch.tensor(self.scales["y_std"]).to(device)

        return data

class HeteroDEMDataset(InMemoryDataset):
    def __init__(self,dataset_name, 
                 dataset_type: Literal["train","validate","test"],
                 root = None, transform = None, 
                 pre_transform = T.Compose([T.ToUndirected(),CartesianHetero(),DistanceHetero()]), 
                 force_reload = False, super_tol=6,
                 push_forward_step_max: int = 0,
                 bundle_size: int = 1,
                 model = None,
                 model_ident:str=None,
                 overfit_sim_idx:int=None,
                 overfit_time_idx:int=None):
        
        root: str = os.path.join(os.getcwd(),"Data")
        self.dataset_type = dataset_type
        self.raw_data_path = os.path.join(root,"raw")
        self.force_reload = force_reload
        self.dataset_name = dataset_name
        self.scale_name = f"{dataset_name}_Hetero"
        self.processed_data_path = os.path.join(root,"processed",dataset_name)
        self.super_tol=super_tol
        self.forward_step_max = push_forward_step_max
        self.bundle_size = bundle_size
        self.model = model
        self.model_ident = model_ident
        self.overfit_sim_idx = overfit_sim_idx
        self.overfit_time_idx = overfit_time_idx

        super().__init__(root, transform, pre_transform,force_reload=force_reload)
        self.load(self.processed_file_names[0])

    def download(self):
        pass

    @property 
    def raw_file_names(self):
        return[f"{self.dataset_name}_Data.npy",
               f"{self.dataset_name}_Topology.npy",
               f"{self.dataset_name}_BC.npy"]
    
    @property
    def processed_file_names(self):
        if self.forward_step_max == 0:
            processed_file_name = f"{self.dataset_name}_het_bund{self.bundle_size}_push{self.forward_step_max}_{self.dataset_type}.pt"
        else:
            processed_file_name = f"{self.dataset_name}_het_bund{self.bundle_size}_push{self.forward_step_max}_{self.model_ident}_{self.dataset_type}.pt"
        return [os.path.join(self.processed_data_path,processed_file_name)]

    def LoadSimTop(self,i):
        data = np.load(os.path.join(self.raw_data_path,self.raw_file_names[i]),allow_pickle=True)
        type_idx = {"train":0,"validate":1,"test":2}[self.dataset_type]
        mask =  DataMask(data)[type_idx]
        return data[mask]

    def SliceAndReshapeData(self,par_data,t,push_forward_steps):
        pos_slice = par_data[t+push_forward_steps*self.bundle_size:t+(push_forward_steps+1)*self.bundle_size,:,:3]
        pos_slice_2D = np.concatenate([pos for pos in pos_slice],axis=1)
        return pos_slice_2D
    
    def GetPushForwardSteps(self):
        if self.forward_step_max == 0:
            push_forward_steps = 0
        else:
            push_forward_steps = np.random.randint(1,self.forward_step_max+1)
        return push_forward_steps
    
    def process(self):
        data_list = []
        data_agr,bc_agr= [self.LoadSimTop(i) for i in [0,2]]
        if self.forward_step_max > 0:
            transform = T.Compose([T.ToUndirected(),CartesianHetero(False),DistanceHetero(),NormalizeHeteroData(self.dataset_name,self.scale_name,edge_only=False)])
            Simulation = LearnedSimulatorHetero(self.model,scale_function=Rescale(self.dataset_name,self.scale_name),transform=transform)
            self.Rollout_step = Simulation.Rollout_Step

        for i, (sim_data, bc) in tqdm(enumerate(zip(data_agr,bc_agr)),total=bc_agr.shape[0]):
            if i is not self.overfit_sim_idx and self.overfit_sim_idx is not None:
                continue

            self.super_topology = ConstructTopology(sim_data[0],bc,self.super_tol)
            if self.forward_step_max > 0: Simulation.super_topology = self.super_topology
            for t in range(len(sim_data)-1*self.bundle_size*(self.forward_step_max+1)):

                if t is not self.overfit_time_idx and self.overfit_time_idx is not None:
                    continue

                par_data = sim_data[t].copy()
                bc_t = bc.copy()
                bc_t[0] = bc[0]+(t+1)*bc[1]

                push_forward_steps = self.GetPushForwardSteps()
                matlab_topology = TopologyFromPlausibleTopology(self.super_topology,par_data,bc_t,0)

                with torch.inference_mode():
                    for forward_step in range(push_forward_steps):
                        par_data, bc_t, matlab_topology = self.Rollout_step(par_data, bc_t, matlab_topology)
                
                bc_t[0] += bc_t[1]

                pos_slice        = self.SliceAndReshapeData(sim_data,t  ,push_forward_steps)
                pos_target_slice = self.SliceAndReshapeData(sim_data,t+1,push_forward_steps)
                pos_slice[:,:3] = par_data[:,:3]
                displacements = pos_target_slice-pos_slice

                data = ToHeteroData(par_data,matlab_topology,bc_t,displacements)
                data.push_forward_steps=push_forward_steps
                data.t = t+push_forward_steps
                data_list.append(data)
        
        print(f"Pre-processing data:")
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        print(f"Normalizing {self.dataset_type} data")    
        if self.dataset_type == "train" and self.forward_step_max == 0:
            GetScales(Batch.from_data_list(data_list),self.dataset_name,self.scale_name,hetero=True)
        self.normalize = NormalizeHeteroData(self.dataset_name,self.scale_name,edge_only=False)
        data_list = [self.normalize(data) for data in tqdm(data_list)]

        self.save(data_list, self.processed_file_names[0])

from ML_functions import SaveModelInfo,SaveTrainingInfo
class TrainHetero():
    def __init__(self,dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers):
        self.dataset_name = dataset_name
        self.model_ident = model_ident
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.msg_num = msg_num
        self.emb_dim = emb_dim
        self.num_layers = num_layers

    def __call__(self,dataset_train,dataset_val=None,retrain:bool=False):
        model,msg = GetHeteroModel(self.dataset_name,self.model_ident,dataset_train[0].metadata(),
                                self.msg_num,self.emb_dim,self.num_layers,retrain)
        
        if msg == 'Loaded model' and retrain == True:
            raise Exception('pre-trained model already exists')
        
        SaveModelInfo(model,self.dataset_name,self.model_ident,hetero=True)
        trainer = HeteroTrainer(model,self.batch_size,self.lr,self.epochs,self.dataset_name,self.model_ident)
        trainer.train_loop(dataset_train,dataset_val)
        SaveTrainingInfo(dataset_train,trainer)

from ML_functions import Rescale
class ForwardTrainHetero():
    def __init__(self,dataset_name:str,model_ident:str,dataset_clean,batch_size:int,lr:float,epochs:int,bundle_size:int):
        self.dataset_name = dataset_name
        self.model_ident = model_ident
        self.dataset_clean = dataset_clean
        self.model_metadata = dataset_clean[0].metadata()
        self.bundle_size = bundle_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.scale_function = Rescale(dataset_name,scale_name=f"{dataset_name}_Hetero")

    def ValidateNoisyDataEquality(self):
        for data_noisy in self.dataset_noisy:
            data_clean = self.dataset_clean[data_noisy.t.item()]

            if data_clean.t.item()!=data_noisy.t.item():
                raise Exception('timestep does not match')
            eqpos_clean = data_clean['particle'].pos.detach().clone()
            eqpos_noisy = data_noisy['particle'].pos.detach().clone()
            eqpos_clean += self.scale_function(data_clean['particle'].y.detach().clone())
            eqpos_noisy += self.scale_function(data_noisy['particle'].y.detach().clone())
            all_eqpos_match = np.all(np.isclose(eqpos_clean,eqpos_noisy,atol=0,rtol=1e-9))

            if all_eqpos_match is False:
                raise Exception('Error in noise injection equilibrium positions')
        print('Noisy labels sucessfully validated')

    def GetModel(self,push_idx):
        if push_idx == 0:
            model,msg = GetHeteroModel(self.dataset_name,self.model_ident,self.model_metadata)
        else:
            model,msg = GetHeteroModel(self.dataset_name,f"{self.model_ident}_Push",self.model_metadata)

        if msg != 'Loaded model':
            raise Exception('Failed to load pre-trained model')
        
        return model
    
    def AugmentDataset(self,model,push_forward_step_max):
        self.dataset_noisy = HeteroDEMDataset(self.dataset_name,'train',
                                                force_reload=True,
                                                bundle_size=self.bundle_size,
                                                model=model,
                                                model_ident=self.model_ident,
                                                overfit_sim_idx=self.dataset_train_clean.overfit_sim_idx,
                                                overfit_time_idx=self.dataset_train_clean.overfit_time_idx,
                                                push_forward_step_max=push_forward_step_max)
        
        data_list_clean = [data for data in self.dataset_clean]
        data_list_noisy = [data for data in self.dataset_noisy]
        dataset = InMemoryDataset()
        dataset.data, dataset.slices = dataset.collate(data_list_clean+data_list_noisy)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset,[0.85,0.15])
        return dataset_train, dataset_val
        
    def __call__(self,push_idx:int=0,push_forward_step_max:int=0,validate_eq:bool=False):
        model = self.GetModel(push_idx)
        self.dataset_train,self.dataset_val = self.AugmentDataset(model,push_forward_step_max)

        if validate_eq is True: self.ValidateNoisyDataEquality()

        print(f"Training {self.dataset_name}_{self.model_ident}_Push{push_idx}")
        trainer = HeteroTrainer(model,self.batch_size,self.lr,self.epochs,self.dataset_name,model_ident=f"{self.model_ident}_Push")    
        trainer.train_loop(self.dataset_train,self.dataset_val)
        SaveTrainingInfo(self.dataset_noisy,trainer)

class HeteroConvEdge(torch.nn.Module):
    r"""Adaptation of the HeteroConv wrapper, this version also outputs edge embeddings.

    Args:
        convs (Dict[Tuple[str, str, str], MessagePassing]): A dictionary
            holding a bipartite
            :class:`~torch_geometric.nn.conv.MessagePassing` layer for each
            individual edge type.
        aggr (str, optional): The aggregation scheme to use for grouping node
            embeddings generated by different relations
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"cat"`, :obj:`None`). (default: :obj:`"sum"`)
    """
    def __init__(
        self,
        convs: Dict[EdgeType, MessagePassing],
        aggr: Optional[str] = "mean",
    ):
        super().__init__()

        for edge_type, module in convs.items():
            check_add_self_loops(module, [edge_type])

        src_node_types = {key[0] for key in convs.keys()}
        dst_node_types = {key[-1] for key in convs.keys()}
        if len(src_node_types - dst_node_types) > 0:
            warnings.warn(
                f"There exist node types ({src_node_types - dst_node_types}) "
                f"whose representations do not get updated during message "
                f"passing as they do not occur as destination type in any "
                f"edge type. This may lead to unexpected behavior.")

        self.convs = ModuleDict(convs)
        self.aggr = aggr

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        *args_dict,
        **kwargs_dict,
    ) -> Dict[NodeType, Tensor]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding node feature
                information for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.
            *args_dict (optional): Additional forward arguments of individual
                :class:`torch_geometric.nn.conv.MessagePassing` layers.
            **kwargs_dict (optional): Additional forward arguments of
                individual :class:`torch_geometric.nn.conv.MessagePassing`
                layers.
        """
        out_dict: Dict[str, List[Tensor]] = {}
        node_aggr_dict, edge_emb_dict = {},{}

        for edge_type, conv in self.convs.items():
            src, rel, dst = edge_type

            has_edge_level_arg = False

            args = []
            for value_dict in args_dict:
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    args.append(value_dict[edge_type])
                elif src == dst and src in value_dict:
                    args.append(value_dict[src])
                elif src in value_dict or dst in value_dict:
                    args.append((
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    ))

            kwargs = {}
            for arg, value_dict in kwargs_dict.items():
                if not arg.endswith('_dict'):
                    raise ValueError(
                        f"Keyword arguments in '{self.__class__.__name__}' "
                        f"need to end with '_dict' (got '{arg}')")

                arg = arg[:-5]  # `{*}_dict`
                if edge_type in value_dict:
                    has_edge_level_arg = True
                    kwargs[arg] = value_dict[edge_type]
                elif src == dst and src in value_dict:
                    kwargs[arg] = value_dict[src]
                elif src in value_dict or dst in value_dict:
                    kwargs[arg] = (
                        value_dict.get(src, None),
                        value_dict.get(dst, None),
                    )

            if not has_edge_level_arg:
                continue

            node_msg, edge_emb = conv(*args, **kwargs)

            if dst not in node_aggr_dict:
                node_aggr_dict[dst] = node_msg
            else:
                if self.aggr == 'mean':
                    node_aggr_dict[dst]+=node_msg
                    node_aggr_dict[dst]/=2
            
            edge_emb_dict[edge_type] = edge_emb

        return node_aggr_dict, edge_emb_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'
    
def GetHeteroModel(dataset_name,model_ident,metadata=None,
                   msg_num=3,emb_dim=64,num_layers=2,retrain:bool=False):
    model_name = f"{dataset_name}_{model_ident}"

    metadata = (['particle', 'wallpoint'],
                [('particle', 'PP_contact', 'particle'),
                ('particle', 'PW_contact', 'wallpoint'),
                ('wallpoint', 'rev_PW_contact', 'particle')])
    
    if model_name[-4:] == "Push":
        model_path = os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name[:-5]}")
    else:
        model_path = os.path.join(os.getcwd(),"Models",dataset_name,f"{model_name}")
    
    model_info_path = f"{model_path}_ModelInfo.json"

    if os.path.exists(model_path) and os.path.exists(model_info_path) and retrain==False:   
        with open(model_info_path) as json_file: settings = json.load(json_file)

        model = HeteroDEMGNN(dataset_name,metadata,
                             msg_num=settings["msg_num"],
                             emb_dim=settings["emb_dim"],
                             hidden_dim=settings["hidden_dim"],
                             num_layers=settings["num_layers"])
        model.load_state_dict(torch.load(model_path))
        msg = "Loaded model"
        print(f"{msg} {model_name}")
    else: 
        msg = "No Trained model"
        print(msg)
        model = HeteroDEMGNN(dataset_name,metadata,msg_num,emb_dim,emb_dim,num_layers)

    return model, msg