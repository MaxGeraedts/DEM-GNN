from Encoding import AggregateRawData, save

dataset_name    = "N400_Mono"

data_dir = "/home/20182319/Data"
ArgsAggregation = AggregateRawData(data_dir,dataset_name)
save(dataset_name,*ArgsAggregation)