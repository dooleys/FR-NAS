def get_model_path(model_name,head, optimizer, dataset):
    model_path_dict = {}
    if dataset == "celeba":
        if model_name == "dpn107":
            if head == "CosFace":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/dpn107_CosFace_SGD_0.1/model_333.pth"
            elif head == "MagFace":
                return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/dpn107_MagFace_SGD_0.1/model_666.pth"
        if model_name == "smac_000":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_000/model_333.pth"
        if model_name == "smac_010":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_010/model_333.pth"
        if model_name == "smac_680":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_680/model_333.pth"

    elif dataset == "vggface2":
        if model_name == "dpn107":
            if optimizer == "AdamW":
                return "/work/dlclarge2/sukthank-ZCP_Competition/samuel_exps/FR-NAS/vggface2_train_111/dpn107_CosFace_AdamW_0.001_cosine/Checkpoint_Head_CosFace_Backbone_dpn107_Opt_AdamW_Dataset_CelebA_Epoch_11.pth/Checkpoint_Head_CosFace_Backbone_dpn107_Opt_AdamW_Dataset_CelebA_Epoch_11.pth"
            elif optimizer == "SGD":
                return "/work/dlclarge2/sukthank-ZCP_Competition/samuel_exps/FR-NAS/vggface2_train_111/dpn107_CosFace_SGD_0.1_cosine/Checkpoint_Head_CosFace_Backbone_dpn107_Opt_SGD_Dataset_CelebA_Epoch_11.pth/Checkpoint_Head_CosFace_Backbone_dpn107_Opt_SGD_Dataset_CelebA_Epoch_11.pth"
        if model_name == "rexnet_200":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/rexnet_200.pth"
        if model_name == "smac_301":
            return "/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_vggface2/model_301.pth"

    