import torch
import torch_geometric.transforms as T
from tqdm import tqdm, trange
import numpy as np
import json
import os
from typing import Literal
from ML_functions import DEM_Dataset,LearnedSimulator, NormalizeData, GetModel, Rescale, NormalizePos, MaskTestData, Trainer
from Encoding import NumpyGroupby, ProjectPointsToHyperplane
from IPython.display import clear_output

def GetAllContactpoints(data:object):
    real_edge = data.edge_index[:,data.edge_mask]
    origins = real_edge[0,:]
    destinations = real_edge[1,:]
    midpoints = (data.pos[origins]+data.pos[destinations])/2
    wallpoints = data.pos[~data.mask]
    contactpoints = torch.concatenate((midpoints,wallpoints))
    return contactpoints

def GetContactPerParticle(data,contactpoints):
    Numpar = data.x[data.mask].shape[0]
    ParContactPoints = [[] for i in range(Numpar)]
    ParContactNormals = [[] for i in range(Numpar)]
    
    for i,par_i in enumerate(data.edge_index[1,:]):
        par_i = int(par_i)
        ParContactPoints[par_i].append(contactpoints[i])
        normal_temp = contactpoints[i]-data.pos[par_i]
        ParContactNormals[par_i].append(normal_temp/np.linalg.norm(normal_temp))
    ParContactPoints = [np.array(ParContactPoints[i]) for i in range(len(ParContactPoints))]
    ParContactNormals = [np.array(ParContactNormals[i]) for i in range(len(ParContactNormals))]  
    return ParContactPoints, ParContactNormals

def EffectiveE(data,idx):
    poisson_ratio = data.x[idx,2]
    Youngs_mod = data.x[idx,1]
    return (1-torch.square(poisson_ratio))/Youngs_mod
    
def EffectiveStiffness(data:object):
    """Calculate relative stiffnes Nij for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 1xNcontact
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    E_ij = EffectiveE(data,i)
    E_ij[data.edge_mask]+=EffectiveE(data,j[data.edge_mask])

    R_ij = 1/data.x[i,0] 
    R_ij[data.edge_mask]+=1/data.x[j[data.edge_mask],0]

    Nij = (4/3)*1/(E_ij)*(1/torch.sqrt(R_ij))
    return Nij

def GetGamma(data):
    """Calculate relative displacement for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 3xNcontact
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    R_i = data.x[i,0]
    R_i += data.x[j,0]
    gamma = R_i - torch.norm(data.pos[i]-data.pos[j],dim=1)
    gamma[gamma<0] = 0
    return gamma

def GetContactForce(data):
    """Calculate contact force vectors for all contacts

    Args:
        data (object): Graph describing particle assembly

    Returns:
        array: 3xNcontact array
    """
    i,j = data.edge_index[1,:], data.edge_index[0,:]
    relative_displacement = GetGamma(data)
    Eff_Stiffness = EffectiveStiffness(data)
    Fij_size = Eff_Stiffness*(relative_displacement*torch.sqrt(relative_displacement))
    contactnormal = (data.pos[i]-data.pos[j])
    contactnormal = torch.div(contactnormal.T,torch.norm(contactnormal,dim=1))
    Fij = Fij_size*contactnormal
    return Fij.T

def GetVolumeAndExtremeDims(BC_t, case):
    """Given a boundary condition array at time t, calculate volume

    Args:
        BC (Array): Boundary condition hyperplane points at timestep

    Returns:
        float: Boundary volume
    """
    xmax, xmin = BC_t.item(0,0), BC_t.item(3,0)
    ymax, ymin = BC_t.item(1,1), BC_t.item(4,1)
    zmax, zmin = BC_t.item(2,2), BC_t.item(5,2)
    maxdim = np.array([[xmin,xmax],[ymin,ymax],[zmin,zmax]])
    vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
    return vol,maxdim

def GetStressTensor(data,BC):
    """Calculate Internal stress tensor for one timestep

    Args:
        data (Data object): Graph describing particle assembly at time t
        BC (array): Boundary condition at time t

    Returns:
        array: 3x3 Gauchy stress tensor
    """
    contactpoints = GetAllContactpoints(data)
    contactvector = contactpoints - data.pos[data.edge_index[1,:]]
    #contactvector = data.pos[data.edge_index[1,:]] - data.pos[data.edge_index[0,:]]

    contactforce = GetContactForce(data)
    stress_tensor = torch.zeros((3,3))
    vol = GetVolumeAndExtremeDims(BC,case='box')[0]
    for contact in range(contactforce.shape[0]):
        Fij = contactforce[contact].reshape(3,1)
        Lij = contactvector[contact].reshape(1,3)
        stress_tensor += Fij*Lij
    stress_tensor /= vol
    return stress_tensor

from Encoding import ConvertToDirected
def GetInternalStressRollout(Rollout):
    """Calculate internal stress tensor (Gauchy) for every timestep

    Args:
        Rollout (list): List of Torch Data objects describing particle assemblies

    Returns:
        array: [Timestep[3x3]]
    """
    stress_evo = torch.zeros((Rollout.timesteps,3,3))
    for t in range(Rollout.timesteps):
        data = Rollout.GroundTruth[t]
        data = ConvertToDirected(data.clone())
        BC = Rollout.BC_rollout[t][0][:,:3]
        stress_evo[t] = GetStressTensor(data,BC)
    return stress_evo

def AggregateForces(datalist):
    """Calculates resultant forces for each particles, their norms and the sum of all norms

    Args:
        datalist (list): Rollout list of data describing graphs

    Returns:
        Farg (tuple): F_agr: resultant force for each particle, F_norm: norms of resultant forces, F_sum: sum of all norms
    """
    F_res = torch.zeros([datalist.shape[0],datalist[0].mask.sum(),3])
    F_contact = [None]*datalist.shape[0]
    for t,data in tqdm(enumerate(datalist)):
        F_contact[t] = GetContactForce(data)
        for i,par_index in enumerate(data.edge_index[1,:]):
            F_res[t,par_index,:] += F_contact[t][i,:]
    F_norm = torch.norm(F_res,dim=2)
    F_sum = torch.sum(F_norm,1)
    return F_contact,F_res, F_norm, F_sum

def GetWallArea(BC,case):
    maxdims = GetVolumeAndExtremeDims(BC,case)[1]
    [x,y,z] = maxdims[:,1] - maxdims[:,0]
    x,y,z = x.item(),y.item(),z.item()
    A_wall = [[y*z,x*z,x*y,y*z,x*z,x*y]]
    return np.array(A_wall)

def GetWallForce(data):
    F_wall = np.zeros((6,3))
    Fcontact = GetContactForce(data).numpy()
    F_PWcontact = Fcontact[np.invert(data.edge_mask),:]

    Wallcontact = data.MatlabTopology[data.MatlabTopology[:,1]<0]
    Wallcontact[:,1] += 1
    Wallcontact[:,1] *= -1
    Wall_idx = Wallcontact[:,1]

    for i,Wall in enumerate(Wall_idx):
        F_wall[Wall,:] += F_PWcontact[i,:]

    return F_wall

def GetWallStress(datalist,BC_rollout,case='box'):
    A_wall = np.array([np.squeeze(GetWallArea(BC[0],case)) for BC in BC_rollout])
    F_wall = np.array([np.linalg.norm(GetWallForce(data),axis=1) for data in datalist])
    S_wall = F_wall/A_wall
    return S_wall

def NormalizedResultantForce(data):
    force = GetContactForce(data).numpy()
    contact = data.edge_index.T.numpy()

    key_sort,force_grouped = NumpyGroupby(group_key=contact[:,1],group_value=force)

    unique_keys = np.ndarray.astype(np.unique(key_sort),int)
    num_particles = data.mask.sum().item()
    force_grouped_indexed = [np.zeros((1,3))]*num_particles
    for i,index in enumerate(unique_keys):
        force_grouped_indexed[index] = force_grouped[i]

    Fres_vectors = np.array([np.sum(group,axis=0) for group in force_grouped_indexed])
    Fres_norm = np.linalg.norm(Fres_vectors,axis=1)

    F_vectors_norm = [np.linalg.norm(group,axis=1,keepdims=True) for group in force_grouped_indexed]
    F_vectors_norm_sum = np.array([np.sum(group).item() for group in F_vectors_norm])
    F_vectors_norm_sum[F_vectors_norm_sum==0] = 1

    Fres_size_normalized  = Fres_norm/F_vectors_norm_sum
    return Fres_size_normalized

from HeteroML import GetHeteroModel,LearnedSimulatorHetero, CartesianHetero,DistanceHetero, NormalizeHeteroData
def AggregatedRollouts(model,AggregatedArgs:tuple,test_dataset_name=None,device:str='cuda'):
    scale_name = f"{test_dataset_name}_Hetero"
    transform = T.Compose([T.ToUndirected(),CartesianHetero(False),DistanceHetero(),NormalizeHeteroData(test_dataset_name,scale_name,edge_only=False,model_ident=model.model_ident)])
    Simulation = LearnedSimulatorHetero(model, scale_function = Rescale(test_dataset_name,model.model_ident,scale_name),transform = transform,device='cpu')

    datalist_ML = []
    datalist_GT = []
    bc_agr      = []
    for sample_idx in trange(AggregatedArgs[0].shape[0]):
        Simulation.Rollout(*AggregatedArgs,sample_idx)
        datalist_ML.append(Simulation.ML_rollout)
        datalist_GT.append(Simulation.GroundTruth)
        bc_agr.append(Simulation.BC_rollout)
    bc_agr = np.concatenate(bc_agr,axis=0)
    return datalist_ML, datalist_GT,bc_agr

def DatalistToArray(datalist):
    pos_array = np.array([[data.pos[data.mask] for data in simulation] for simulation in datalist])
    property_array = np.array([[data.x[data.mask,:3] for data in simulation] for simulation in datalist])
    return pos_array, property_array

def GeometricMetrics(pos_test,pos_ML,radii):
    [pos_test,pos_ML,radii] = [torch.from_numpy(arg) for arg in [pos_test,pos_ML,radii]]

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(pos_ML,pos_test)

    dist = (pos_ML-pos_test).pow(2).sum(dim=-1).sqrt()
    dist_mean = dist.mean()

    normalized_dist = dist/radii
    normalized_dist_mean = normalized_dist.mean()
    return loss.item(), dist_mean.item(), normalized_dist_mean.item()

class Evaluation:
    def __init__(self,mode:Literal["geometric","mechanics_sum","mechanics_mean"],print_results:bool=False,show_tqdm = False):
        self.mode = mode
        self.print_results = print_results
        self.show_tqdm = show_tqdm
        self._disable_tqdm = not show_tqdm

        if mode == "mechanics_sum":
            self.aggregation_function = np.sum
            self.description = "Mean sum of normalized resultantant forces"
            self.abreviation = "MSNRF"
        if mode == "mechanics_mean":
            self.aggregation_function = np.mean
            self.description = "Mean of normalized resultantant forces"
            self.abreviation = "MNRF"
        else:
            self.description = "Geometric measures"

    def EvaluateGeometric(self,datalist_ML,datalist_GT):
        pos_ML = DatalistToArray(datalist_ML)[0]
        pos_GT, prop_GT = DatalistToArray(datalist_GT)
        radii = prop_GT[:,:,:,0]

        loss, dist_mean, normalized_dist_mean = GeometricMetrics(pos_GT,pos_ML,radii)

        metrics = {'L2 Norm:': loss, 
                   "Mean Euclidean distance:": dist_mean, 
                   "Radius normalized mean Euclidean distance:": normalized_dist_mean}
        
        return metrics
    
    def EvaluateMechanics(self,datalist_ML,datalist_GT):
        #CHECK NORMALIZATION!
        metrics = {}
        if datalist_GT is not None:
            ground_truth    = np.array([[self.aggregation_function(NormalizedResultantForce(data)) for data in datalist_sample] for datalist_sample in tqdm(datalist_GT,disable=self._disable_tqdm)])
            metrics["Ground truth"] = np.mean(ground_truth).item()

        model_prediction= np.array([[self.aggregation_function(NormalizedResultantForce(data)) for data in datalist_sample] for datalist_sample in tqdm(datalist_ML,disable=self._disable_tqdm)])
        metrics["Model:"] = np.mean(model_prediction).item()
        return metrics

    def __call__(self,datalist_ML,datalist_GT):
        if self.mode == "geometric":
            metrics = self.EvaluateGeometric(datalist_ML,datalist_GT)
        else:
            metrics = self.EvaluateMechanics(datalist_ML,datalist_GT)

        if self.print_results == True:
            [print(f"{metric:<50}{value:4f}") for metric, value in metrics.items()]   

        return metrics 
       
class MSEloss(Trainer):
    def __init__(self,model,batch_size):
        super().__init__(model, dataset_name=None, model_ident=None, batch_size=batch_size,lr=None, epochs=None)

    def __call__(self,dataset_test):
        test_dl = self.make_data_loader(dataset_test,False)
        mean_test_loss = self.batch_loop(test_dl,disable_tqdm=False)[0]
        return mean_test_loss 
       
class CompareModels():
    def __init__(self,test_dataset_name:str, model_dataset_name:str,model_ident,save_name,evaluation_function=Evaluation(mode='mechanics_mean',print_results=True)):
        try:
            filename = os.path.join(".",'Evaluation',f"{save_name}_metrics.json")
            with open(filename, 'r') as file:
                self.metric_dict = json.load(file)
        except:
            self.metric_dict = {}
        self.AggregatedArgs = MaskTestData(test_dataset_name,"test")
        self.test_dataset_name = test_dataset_name
        self.model_dataset_name = model_dataset_name
        self.model_ident = model_ident
        self.eval_function = evaluation_function
        self.model = GetModel(model_dataset_name,model_ident)[0]
        AggregatedArgs = MaskTestData(test_dataset_name,"test")
        self.datalist_ML, self.datalist_GT, self.bc_agr = AggregatedRollouts(self.model,AggregatedArgs,test_dataset_name)
        self.save_name = save_name

    def EvaluateEquilibrium(self,eval_GT:bool=False):
        if eval_GT is True:
            metrics = self.eval_function(self.datalist_ML, self.datalist_GT)
        else:
            metrics = self.eval_function(self.datalist_ML, None)

        for (key,value) in metrics.items():
            if key == "Model:":
                self.metric_dict[f"{self.eval_function.abreviation}: {self.model_ident}"] = value
            else:
                self.metric_dict['MNRF: Groundtruth'] = value

    def EvaluateMSE(self,batch_size):   
        dataset_test = DEM_Dataset(self.test_dataset_name, 'test', force_reload=False, bundle_size=self.model.bundle_size)
        loss_function = MSEloss(self.model,batch_size)
        MSE_loss = loss_function(dataset_test)
        self.metric_dict[f"MSE: {self.model_ident}"] = MSE_loss

    def EvaluateParticlesOutsideBoundary(self):
        data_list_flat = [data for datalist in self.datalist_ML for data in datalist]
        par_bool = ParticlesOutsideBoundary(data_list_flat,self.bc_agr)
        outpar = np.mean(par_bool)
        self.metric_dict[f"Escaping Particles: {self.model_ident}"] = outpar.item()
        return par_bool
    
    def PrintResults(self):
        print("\n",self.eval_function.description,"\n") 
        for metric, value in self.metric_dict.items():
            if self.eval_function.abreviation in metric: print(f"{metric:<50}{value:.3f}")

        print("\n","One-step Mean Squared Error","\n")
        for metric, value in self.metric_dict.items():
            if "MSE" in metric: print(f"{metric:<50}{value:.3f}")
        
        print("\n","Mean number of particles escaping Boundary Conditions","\n")
        for metric, value in self.metric_dict.items():
            if "Escaping Particles" in metric: print(f"{metric:<50}{value:.3f}")

    def SaveResults(self):
        filename = os.path.join(".",'Evaluation',f"{self.save_name}_metrics.json")
        with open(filename,'w') as f:
            json.dump(self.metric_dict,f)
    
def CoordinationNumber(datalist):
    contact_num = np.array([data.edge_index.shape[1] for data in datalist])
    par_num = np.array([data.mask.sum().item() for data in datalist])
    coordination_number = contact_num/par_num
    return coordination_number

def CheckCylindricalBC(points,cyl):
    point_to_cyl_origin = points-cyl[:3]
    cyl_axis = cyl[3:6]/np.linalg.norm(cyl[3:6])
    radius = cyl[6]

    axis_projection = point_to_cyl_origin*cyl_axis*cyl_axis
    axis_projection += cyl[:3]
    dist_axis = np.linalg.norm(points-axis_projection,axis=1)

    if cyl[-2] == 1: 
        bc_bool = dist_axis <= radius
    if cyl[-2] == -1:
        bc_bool = dist_axis >= radius

    return bc_bool 

def CheckPlanarBC(points,plane):
    wallpoints = ProjectPointsToHyperplane(points,plane)  
    vector = points-wallpoints 
    vector /= np.linalg.norm(vector,axis=1,keepdims=True) 
    normal_vector = plane[3:6]
    bc_bool = np.all(vector == normal_vector,axis=1)    
    return bc_bool

def ValidateBC(par,bc_step):
    bc_bool = np.empty([par.shape[0],bc_step.shape[1]])
    for wall_id,wall in enumerate(bc_step[0]):
        if wall[-1] == 1:
            bc_bool[:,wall_id] = CheckCylindricalBC(par,wall)
        if wall[-1] == 0:
            bc_bool[:,wall_id] = CheckPlanarBC(par,wall)
    return bc_bool

def ParticlesOutsideBoundary(data_list,bc_rollout):
    outside_particles = np.zeros(len(data_list))
    for t,(data,bc_step) in enumerate(zip(data_list,bc_rollout)):
        bc_bool = ValidateBC(np.array(data.pos[data.mask]),np.array(bc_step))
        par_in_bounds = np.all(bc_bool,axis=1)
        num_par_out_bounds = np.sum(~par_in_bounds).item()
        outside_particles[t] = num_par_out_bounds
    return outside_particles

def EvaluateExperiment(exp_settings,save_name:str,batch_size:int):
    par_bool = []
    for i,(exp_args) in enumerate(exp_settings):
        eval = CompareModels(*exp_args,save_name)
        if i == 0:eval_GT=True
        eval.EvaluateEquilibrium(eval_GT)
        eval.EvaluateMSE(batch_size)
        par_bool.append(eval.EvaluateParticlesOutsideBoundary())
        if save_name is not None: eval.SaveResults()
        clear_output()
    eval.PrintResults()
    return eval.metric_dict

def GetExperimentSettings(experiment_ident:Literal['N400Embedding','2Sphere','N400msg','N400Error','N400layer'], test_dataset_name:str,push:str=False,input_list:list=[]):

    if experiment_ident == 'N400Embedding':
        model_idents = ['Emb16','Emb32','Emb64','Emb128','Emb256']
        model_dataset_names = ['N400_Mono']*len(model_idents)
        test_dataset_names = [test_dataset_name]*len(model_idents)

    if experiment_ident == 'N400Error':
        model_idents = ['Emb128','bundle','forward5','forward10','forward15','forward20','Allout']
        model_dataset_names = ['N400_Mono']*len(model_idents)
        test_dataset_names = [test_dataset_name]*len(model_idents)

    if experiment_ident == 'N400msg':
        model_idents = [f'msg{i}' for i in input_list]
        model_dataset_names = ['N400_Mono']*len(model_idents)
        test_dataset_names = [test_dataset_name]*len(model_idents)
    
    if experiment_ident == 'N400layer':
        model_idents = ['layer1','layer2','Allout','layer4']
        model_dataset_names = ['N400_Mono']*len(model_idents)
        test_dataset_names = [test_dataset_name]*len(model_idents)

    if experiment_ident == '2Sphere':
        model_idents = ['redo','Allout','lr_small']
        model_dataset_names = ['2Sphere']*len(model_idents)
        test_dataset_names = [test_dataset_name]*len(model_idents)
    
    if push == True:
        model_idents = [string+'_Push' for string in model_idents]
    
    exp_settings = np.array([test_dataset_names,model_dataset_names,model_idents]).T
    return exp_settings