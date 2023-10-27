def get_model_path(model_name,head, optimizer, dataset):
    model_path_dict = {}
    if dataset == "celeba":
        if model_name == "dpn107":
            if head == "CosFace":
                return "models/celeba/model_dpn_cosface_sgd.pth"
            elif head == "MagFace":
                return "models/celeba/model_dpn_magface_sgd.pth"
        if model_name == "smac_000":
            return "models/celeba/smac_000.pth"
        if model_name == "smac_010":
            return "models/celeba/smac_010.pth"
        if model_name == "smac_680":
            return "models/celeba/smac_680.pth"

    elif dataset == "vggface2":
        if model_name == "dpn107":
            if optimizer == "AdamW":
                return "models/vggface2/dpn_cosface_adamw.pth"
            elif optimizer == "SGD":
                return "models/vggface2/dpn_cosface_sgd.pth"
        if model_name == "rexnet_200":
            return "models/vggface2/rexnet_200.pth"
        if model_name == "smac_301":
            return "models/vggface2/smac_301.pth"

    