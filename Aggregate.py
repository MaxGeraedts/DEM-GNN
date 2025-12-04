from Encoding import AggregateRawData, save, GetDataDir
from HeteroML import HeteroDEMDataset
from Evaluation import AverageDEMruntime
dataset = False
aggregate = False
time = True

dataset_name = r"2Sphere"
data_dir = GetDataDir()

if aggregate == True:
    par_data, topology, boundary_conditions, = None, None, None
    Aggregate = AggregateRawData(dataset_name,data_dir)
    Aggregate.DetectShortSim(100)
    par_data = Aggregate.ParticleData()
    #topology = Aggregate.TopologyData()
    boundary_conditions = Aggregate.BoundaryConditions()

    save(dataset_name,par_data,topology,boundary_conditions)

if dataset == True:
    HeteroDEMDataset(dataset_name,force_reload=True)

if time ==True:
    runtime = AverageDEMruntime(dataset_name,data_dir,packing=False)
    mean_runtime, runtimes =runtime()
    print(mean_runtime)
