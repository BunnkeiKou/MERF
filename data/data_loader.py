
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    # 读入图片，生成data loader
    data_loader.initialize(opt)
    return data_loader
