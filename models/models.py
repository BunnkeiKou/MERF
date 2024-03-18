
def create_model(opt):
    model = None
    print(opt.model)
    from .single_model import SingleModel
    model = SingleModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
