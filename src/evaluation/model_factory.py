
def get_model(model_name, dataset):
    raise NotImplementedError


def get_model_path(model_name, dataset):
    model_path_dict = {}
    if dataset == "celeba":
        model_path_dict["dpn107_CosFace_SGD"] = "checkpoints_celeba/densenet161_CosFace_Adam/model_444.pth"
        model_path_dict["dpn107_MagFace_SGD"] = "checkpoints_celeba/dpn107_CosFace_SGD_0.1/model_444.pth"
        model_path_dict["ese_vovnet39b_CosFace"] = "checkpoints_celeba/dpn107_MagFace_SGD_0.1/model_444.pth"
        model_path_dict["mobilenetv3_large_100"] = "checkpoints_celeba/ese_vovnet39b_CosFace_AdamW/model_444.pth"
        model_path_dict["rexnet_200"] = "checkpoints_celeba/rexnet_200_MagFace_SGD/model_444.pth"
        model_path_dict["model_000"] = "checkpoints_celeba/model_000/model_444.pth"
        model_path_dict["model_010"] = "checkpoints_celeba/model_010/model_444.pth"
        model_path_dict["model_680"] = "checkpoints_celeba/model_680/model_444.pth"

    elif dataset == "vggface2":
        model_path_dict["dpn107_CosFace_Adamw"] = "checkpoints_vggface2/dpn_cosface_adamw.pth"
        model_path_dict["dpn107_CosFace_SGD"] = "checkpoints_vggface2/dpn_cosface_sgd.pth"
        model_path_dict["rexnet_200"] = "checkpoints_vggface2/rexnet_200.pth"
        model_path_dict["swinir"] = "checkpoints_vggface2/swinir.pth"
        model_path_dict["model_301"] = "checkpoints_vggface2/model_301.pth"
    
    else:
        raise NotImplementedError
    return model_path_dict[model_name]


def get_model_last_layer_dim(model_name, dataset):
    model_dim_dict = {}
    if dataset == "celeba":
        model_dim_dict["dpn107_CosFace_SGD"] = 1000
        model_dim_dict["dpn107_MagFace_SGD"] = 1000
        model_dim_dict["ese_vovnet39b_CosFace"] = 1000
        model_dim_dict["mobilenetv3_large_100"] = 1000
        model_dim_dict["rexnet_200"] = 1000
        model_dim_dict["model_000"] = 1000
        model_dim_dict["model_010"] = 1000
        model_dim_dict["model_680"] = 1000

    elif dataset == "vggface2":
        model_dim_dict["dpn107_CosFace_Adamw"] = 1000
        model_dim_dict["dpn107_CosFace_SGD"] = 1000
        model_dim_dict["rexnet_200"] = 1000
        model_dim_dict["swinir"] = 1000
        model_dim_dict["model_301"] = 1000
    
    else:
        raise NotImplementedError
    return model_dim_dict[model_name]
    