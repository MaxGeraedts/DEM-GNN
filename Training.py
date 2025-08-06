import torch.cuda
import torch_geometric.transforms as T
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo
import os
import numpy as np

print(torch.cuda.is_available())

force_reload    = True
train           = True
dataset_name    = "N400_Mono"
model_ident     = "lr1"
bundle_size     = 1 
forward_steps   = 0
msg_num         = 3
emb_dim         = 16
learning_rate   = 1
batch_size      = 32
epochs          = 200
pre_transform = T.Compose([T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])

try:
    os.mkdir(os.path.join(os.getcwd(),"Data","processed",dataset_name))
except OSError as e:
    print("Error:", e)


[dataset_train, dataset_val, dataset_test]      = [DEM_Dataset(dataset_name,
                                                               dataset_type,
                                                               force_reload     = force_reload,
                                                               pre_transform    = pre_transform,
                                                               bundle_size      = bundle_size) 
                                                               for dataset_type in ["train","validate","test"]]

if train == True:
    try: 
        os.mkdir(os.path.join(os.getcwd(),"Models",dataset_name))
    except OSError as e:
        print("Error:", e)
    
    model_name=f"{dataset_name}_{model_ident}"
    model, msg = GetModel(dataset_name,
                          model_ident,
                          msg_num=msg_num,
                          emb_dim=emb_dim,
                          num_layers=3,
                          bundle_size=bundle_size)
    SaveModelInfo(model,dataset_name,model_ident)
    
    if msg == "No Trained model":
        print(f"Training {model_name}")
        trainer = Trainer(model, dataset_train,dataset_val,batch_size,learning_rate,epochs,dataset_name,model_ident)
        trainer.train_loop()
        SaveTrainingInfo(dataset_train,trainer)

    model, msg = GetModel(dataset_name,model_ident)
    
if train == True and msg == 'Loaded model':
    if forward_steps == 0: force_reload = False
    [dataset_train]      = [DEM_Dataset(dataset_name,
                                        dataset_type,
                                        force_reload            = force_reload,
                                        pre_transform           = pre_transform,
                                        push_forward_step_max   = forward_steps,
                                        bundle_size             = bundle_size,
                                        model = model,
                                        model_ident = model_ident) 
                                        for dataset_type in ["train"]]
    print(f"Training {model_name}_Push")
    trainer = Trainer(model, dataset_train,dataset_val,batch_size,learning_rate,epochs,dataset_name,model_ident=f"{model_ident}_Push")    
    trainer.train_loop()
    SaveTrainingInfo(dataset_train,trainer)