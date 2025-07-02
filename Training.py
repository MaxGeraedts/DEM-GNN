import torch.cuda
import torch_geometric.transforms as T
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo

print(torch.cuda.is_available())

force_reload    = False
train           = True
dataset_name    = "2Sphere"
model_ident     = "NewModel_2"

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
                          msg_num=2,
                          emb_dim=32,
                          edge_dim=4,
                          num_layers=2)
    SaveModelInfo(model,dataset_name,model_ident)
    
    if msg == "No Trained model":
        trainer = Trainer(model, dataset_train,dataset_val,
                        batch_size=32,
                        lr=0.0000001,
                        epochs=250,
                        model_name=model_name)
        
        trainer.train_loop()
        SaveTrainingInfo(dataset_train,trainer)

    model, msg = GetModel(model_name)
    
if train == True and msg == 'Loaded model':
    [dataset_train]      = [DEM_Dataset(dataset_name,
                                        dataset_type,
                                        mode             = 'delta',
                                        force_reload     = force_reload,
                                        pre_transform    = pre_transform,
                                        super_tol        = 6,
                                        tol              = 0,
                                        noise_factor     = 0,
                                        push_forward_step_max=8,
                                        model = model) 
                                        for dataset_type in ["train"]]
    trainer = Trainer(model, dataset_train,dataset_val,
                      batch_size=32,
                      lr=0.0000001,
                      epochs=250,
                      model_name=f"{dataset_name}_{model_ident}_Push")    
    trainer.train_loop()
    SaveTrainingInfo(dataset_train,trainer)