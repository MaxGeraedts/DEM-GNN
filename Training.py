import torch.cuda
import torch_geometric.transforms as T

from Encoding import AggregateRawData, save
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo

print(torch.cuda.is_available())

aggregate       = False
force_reload    = True
train           = True
dataset_name    = "N400_Mono"
model_ident     = "NewModel_1"

if aggregate == True:
    data_dir = "/home/20182319/Data"
    ArgsAggregation = AggregateRawData(data_dir,dataset_name)
    save(dataset_name,*ArgsAggregation)

pre_transform = T.Compose([T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])

[dataset_train, dataset_val, dataset_test]      = [DEM_Dataset(dataset_name,
                                                               dataset_type,
                                                               mode             = 'delta',
                                                               force_reload     = force_reload,
                                                               pre_transform    = pre_transform,
                                                               super_tol        = 6,
                                                               tol              = 0,
                                                               noise_factor     = 0) 
                                                               for dataset_type in ["train","validate","test"]]

if train == True:
    model = GetModel(dataset_name,model_ident,
                     msg_num=8,
                     emb_dim=64,
                     edge_dim=4,
                     num_layers=2)
    
    SaveModelInfo(model,dataset_name,model_ident)
    
    trainer = Trainer(model, dataset_train,dataset_val,
                      batch_size=32,
                      lr=0.0000001,
                      epochs=1000,
                      model_name=f"{dataset_name}_{model_ident}")
    
    trainer.train_loop()

    SaveTrainingInfo(dataset_train,trainer)