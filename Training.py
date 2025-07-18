import torch.cuda
import torch_geometric.transforms as T
from ML_functions import DEM_Dataset, Trainer, GetModel, SaveModelInfo, SaveTrainingInfo

print(torch.cuda.is_available())

force_reload    = False
train           = True
dataset_name    = "2Sphere"
model_ident     = "Push_Bundle"
bundle_size     = 3 
forward_steps   = 5

pre_transform = T.Compose([T.Cartesian(False),
                           T.Distance(norm=False,cat=True)])

[dataset_train, dataset_val, dataset_test]      = [DEM_Dataset(dataset_name,
                                                               dataset_type,
                                                               force_reload     = force_reload,
                                                               pre_transform    = pre_transform,
                                                               bundle_size      = bundle_size) 
                                                               for dataset_type in ["train","validate","test"]]

if train == True:
    model_name=f"{dataset_name}_{model_ident}"
    model, msg = GetModel(model_name,
                          msg_num=2,
                          emb_dim=32,
                          edge_dim=4,
                          num_layers=2,
                          bundle_size=bundle_size)
    SaveModelInfo(model,dataset_name,model_ident)
    
    if msg == "No Trained model":
        print(f"Training {model_name}")
        trainer = Trainer(model, dataset_train,dataset_val,
                        batch_size=32,
                        lr=0.0000001,
                        epochs=5,
                        model_name=model_name)
        
        trainer.train_loop()
        SaveTrainingInfo(dataset_train,trainer)

    model, msg = GetModel(model_name)
    
if train == True and msg == 'Loaded model':
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
    trainer = Trainer(model, dataset_train,dataset_val,
                      batch_size=32,
                      lr=0.0000001,
                      epochs=5,
                      model_name=f"{dataset_name}_{model_ident}_Push")    
    trainer.train_loop()
    SaveTrainingInfo(dataset_train,trainer)