import torch.cuda
import torch_geometric.transforms as T
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo
import os
import numpy as np

print(torch.cuda.is_available())

force_reload    = False
train           = True
dataset_name    = "N400_Mono"
msg_num         = 3

model_ident     = f"msg{msg_num}"
bundle_size     = 3 
forward_steps   = 5

emb_dim         = 128
learning_rate   = 0.000001
batch_size      = 8
epochs          = 100
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
        trainer = Trainer(model,batch_size,learning_rate,epochs,dataset_name,model_ident)
        trainer.train_loop(dataset_train,dataset_val)
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
    trainer = Trainer(model,batch_size,learning_rate,epochs,dataset_name,model_ident=f"{model_ident}_Push")    
    trainer.train_loop( dataset_train,dataset_val)
    SaveTrainingInfo(dataset_train,trainer)