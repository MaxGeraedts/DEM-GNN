import torch.cuda
import torch_geometric.transforms as T
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo

print(torch.cuda.is_available())

force_reload    = True
train           = True
dataset_name    = "N400_Mono"
model_ident     = "NewModel_1"

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
    model_name=f"{dataset_name}_{model_ident}"
    model, msg = GetModel(model_name,
                          msg_num=8,
                          emb_dim=64,
                          edge_dim=4,
                          num_layers=2)
    
    SaveModelInfo(model,dataset_name,model_ident)
    
    trainer = Trainer(model, dataset_train,dataset_val,
                      batch_size=32,
                      lr=0.0000001,
                      epochs=1000,
                      model_name=model_name)
    
    trainer.train_loop()

    SaveTrainingInfo(dataset_train,trainer)

    model, msg = GetModel(dataset_name,model_ident)
    
if train == True & msg == 'Loaded model':
    [dataset_train, dataset_val, dataset_test]      = [DEM_Dataset(dataset_name,
                                                                   dataset_type,
                                                                   mode             = 'delta',
                                                                   force_reload     = force_reload,
                                                                   pre_transform    = pre_transform,
                                                                   super_tol        = 6,
                                                                   tol              = 0,
                                                                   noise_factor     = 0,
                                                                   push_forward_step_max=4,
                                                                   model = model) 
                                                                   for dataset_type in ["train","validate","test"]]
    trainer = Trainer(model, dataset_train,dataset_val,
                      batch_size=32,
                      lr=0.0000001,
                      epochs=1000,
                      model_name=f"{dataset_name}_{model_ident}_Push")    

    SaveTrainingInfo(dataset_train,trainer)