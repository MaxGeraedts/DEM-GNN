from Encoding import AggregateRawData, save, GetDataDir
from ML_functions import HeteroDEMDataset
import os
dataset = True
aggregate = False

dataset_name = r"N400_Mono"
data_dir = GetDataDir()

if aggregate == True:
    par_data, topology, boundary_conditions, = None, None, None
    Aggregate = AggregateRawData(dataset_name,data_dir)
    par_data = Aggregate.ParticleData()
    #topology = Aggregate.TopologyData()
    boundary_conditions = Aggregate.BoundaryConditions()

    save(dataset_name,par_data,topology,boundary_conditions)

if dataset == True:
    HeteroDEMDataset(dataset_name,force_reload=True)
