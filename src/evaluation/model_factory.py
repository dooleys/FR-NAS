def get_model_path(model_name,head, optimizer, dataset):
    model_path_dict = {}
    if dataset == "celeba":
        if model_name == "dpn107":
            if head == "CosFace":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/densenet161_CosFace_Adam/model_444.pth"
            elif head == "MagFace":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/dpn107_CosFace_SGD_0.1/model_444.pth"
        if model_name == "ese_vovnet39b":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/ese_vovnet39b_CosFace_AdamW/model_444.pth"
        if model_name == "mobilenetv3_large_100":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/mobilenetv3_large_100/model_444.pth"
        if model_name == "rexnet_200":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/rexnet_200_MagFace_SGD/model_444.pth"
        if model_name == "model_000":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_000/model_444.pth"
        if model_name == "model_010":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_010/model_444.pth"
        if model_name == "model_680":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_680/model_444.pth"

    elif dataset == "vggface2":
        if model_name == "dpn107":
            if optimizer == "AdamW":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/dpn_cosface_adamw.pth"
            elif optimizer == "SGD":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/dpn_cosface_sgd.pth"
        if model_name == "rexnet_200":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/rexnet_200.pth"
        if model_name == "swinir":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/swinir.pth"
        if model_name == "model_301":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/model_301.pth"

    