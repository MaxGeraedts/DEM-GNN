from HeteroML import HeteroDEMDataset, HeteroTrainer, HeteroDEMGNN, GetHeteroModel, TrainHetero,ForwardTrainHetero


dataset_name    = 'N050_Mono'
model_ident     = 'Simfit64'
retrain         = True

batch_size      = 1
lr              = 0.01
epochs          = 1000

push_forward_loops = 5
push_forward_epochs = 200
push_forward_step_max_list:list = [20]*push_forward_loops

msg_num = 5
emb_dim = 64
num_layers = 3

dataset_train, dataset_val = [HeteroDEMDataset(dataset_name,dataset_type,force_reload=True,overfit_sim_idx=0) 
                                                for dataset_type in ['train', 'validate']]

train = TrainHetero(dataset_name,model_ident,batch_size,lr,epochs,msg_num,emb_dim,num_layers)
train(dataset_train,retrain)

train_forward = ForwardTrainHetero(dataset_name,model_ident,dataset_train,batch_size,lr,push_forward_epochs,bundle_size=1)

for push_idx, push_forward_step_max in enumerate(push_forward_step_max_list):
    train_forward(push_idx,push_forward_step_max,validate_eq=True)

