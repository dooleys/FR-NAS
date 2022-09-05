"""
This script runs a Multi-Objective Hyperparameter Optimisation using MODEHB to tune the architecture and
training hyperparameters for training a neural network on MNIST in PyTorch. It minimizes two objectives: loss and model size
This example is an extension of single objective problem:'03_pytorch_mnist_hpo.py' to multi-objective setting
Additional requirements:
* torch>=1.7.1
* torchvision>=0.8.2
* torchsummary>=1.5.1

PyTorch code referenced from: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import os
import pickle
import time
from fairness_objective_rexnet import fairness_objective_rexnet
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from distributed import Client
from torchsummary import summary
from torchvision import transforms
from rexnet_graph import  ReXNetV1
from dehb import MODEHB



'''def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)

    # Hyperparameter defining first Conv layer
    kernel1 = CSH.OrdinalHyperparameter("kernel_1", sequence=[3, 5, 7], default_value=5)
    channels1 = CSH.UniformIntegerHyperparameter("channels_1", lower=3, upper=64,
                                                 default_value=32)
    stride1 = CSH.UniformIntegerHyperparameter("stride_1", lower=1, upper=2, default_value=1)
    cs.add_hyperparameters([kernel1, channels1, stride1])

    # Hyperparameter defining second Conv layer
    kernel2 = CSH.OrdinalHyperparameter("kernel_2", sequence=[3, 5, 7], default_value=5)
    channels2 = CSH.UniformIntegerHyperparameter("channels_2", lower=3, upper=64,
                                                 default_value=32)
    stride2 = CSH.UniformIntegerHyperparameter("stride_2", lower=1, upper=2, default_value=1)
    cs.add_hyperparameters([kernel2, channels2, stride2])

    # Hyperparameter for FC layer
    hidden = CSH.UniformIntegerHyperparameter(
        "hidden", lower=32, upper=256, log=True, default_value=128
    )
    cs.add_hyperparameter(hidden)

    # Regularization Hyperparameter
    dropout = CSH.UniformFloatHyperparameter("dropout", lower=0, upper=0.5, default_value=0.1)
    cs.add_hyperparameter(dropout)

    # Training Hyperparameters
    batch_size = CSH.OrdinalHyperparameter(
        "batch_size", sequence=[2, 4, 8, 16, 32, 64], default_value=4
    )
    lr = CSH.UniformFloatHyperparameter("lr", lower=1e-6, upper=0.1, log=True,
                                        default_value=1e-3)
    cs.add_hyperparameters([batch_size, lr])
    return cs'''
def get_configspace(seed=None):
    cs = CS.ConfigurationSpace(seed)
    #width_mult = CSH.OrdinalHyperparameter("width_mult", sequence=[1,1.3,1.5,2.0], default_value=1)
    #ch_div = CSH.OrdinalHyperparameter("ch_div", sequence=[1,8], default_value=1)
    layer1 = CSH.OrdinalHyperparameter("layer1", sequence=[1,2], default_value=1)
    layer2 = CSH.OrdinalHyperparameter("layer2", sequence=[1,2,3], default_value=2)
    layer3 = CSH.OrdinalHyperparameter("layer3", sequence=[1,2,3], default_value=2)
    layer4 = CSH.OrdinalHyperparameter("layer4", sequence=[2,3,4], default_value=3)
    layer5 = CSH.OrdinalHyperparameter("layer5", sequence=[2,3,4], default_value=3)
    layer6 = CSH.OrdinalHyperparameter("layer6", sequence=[4,5,6], default_value=5)
    lr = CSH.UniformFloatHyperparameter("lr", lower=1e-4, upper=0.1, log=True, default_value=1e-3)
    optimizer = CSH.OrdinalHyperparameter("optimizer", sequence=["AdamW","SGD"], default_value="Adam")
    head = CSH.OrdinalHyperparameter("head", sequence=["CosFace","ArcFace","MagFace"], default_value="CosFace")
    cs.add_hyperparameters([layer1,layer2,layer3,layer4,layer5,layer6,lr,optimizer,head])
    return cs





def input_arguments():
    parser = argparse.ArgumentParser(description='Optimizing MNIST in PyTorch using DEHB.')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 123)')
    parser.add_argument('--refit_training', action='store_true', default=False,
                        help='Refit with incumbent configuration on full training data and budget')
    parser.add_argument('--min_budget', type=float, default=1,
                        help='Minimum budget (epoch length)')
    parser.add_argument('--max_budget', type=float, default=20,
                        help='Maximum budget (epoch length)')
    parser.add_argument('--eta', type=int, default=3,
                        help='Parameter for Hyperband controlling early stopping aggressiveness')
    parser.add_argument('--output_path', type=str, default="./pytorch_rexnet_modehb",
                        help='Directory for DEHB to write logs and outputs')
    parser.add_argument('--scheduler_file', type=str, default=None,
                        help='The file to connect a Dask client with a Dask scheduler')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of CPU workers for DEHB to distribute function evaluations to')
    parser.add_argument('--single_node_with_gpus', default=False, action="store_true",
                        help='If True, signals the DEHB run to assume all required GPUs are on '
                             'the same node/machine. To be specified as True if no client is '
                             'passed and n_workers > 1. Should be set to False if a client is '
                             'specified as a scheduler-file created. The onus of GPU usage is then'
                             'on the Dask workers created and mapped to the scheduler-file.')
    mo_strategy_choices = ['EPSNET', 'NSGA-II']
    parser.add_argument('--mo_strategy', default="EPSNET", choices=mo_strategy_choices,
                        type=str, nargs='?',
                        help="specify the multiobjective  strategy from among {}".format(mo_strategy_choices))
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Decides verbosity of DEHB optimization')
    parser.add_argument('--runtime', type=float, default=30000000,
                        help='Total time in seconds as budget to run DEHB')
    args = parser.parse_args()
    return args


def main():
    args = input_arguments()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Get configuration space
    cs = get_configspace(args.seed)
    dimensions = len(cs.get_hyperparameters())

    # Some insights into Dask interfaces to DEHB and handling GPU devices for parallelism:
    # * if args.scheduler_file is specified, args.n_workers need not be specifed --- since
    #    args.scheduler_file indicates a Dask client/server is active
    # * if args.scheduler_file is not specified and args.n_workers > 1 --- the DEHB object
    #    creates a Dask client as at instantiation and dies with the associated DEHB object
    # * if args.single_node_with_gpus is True --- assumes that all GPU devices indicated
    #    through the environment variable "CUDA_VISIBLE_DEVICES" resides on the same machine

    # Dask checks and setups
    single_node_with_gpus = args.single_node_with_gpus
    if args.scheduler_file is not None and os.path.isfile(args.scheduler_file):
        client = Client(scheduler_file=args.scheduler_file)
        # explicitly delegating GPU handling to Dask workers defined
        single_node_with_gpus = False
    else:
        client = None

    ###########################
    # DEHB optimisation block #
    ###########################
    np.random.seed(args.seed)
    modehb = MODEHB(objective_function=fairness_objective_rexnet, cs=cs, dimensions=dimensions, min_budget=args.min_budget,
                    max_budget=args.max_budget, eta=args.eta, output_path=args.output_path,
                    num_objectives=2, mo_strategy=args.mo_strategy,
                    # if client is not None and of type Client, n_workers is ignored
                    # if client is None, a Dask client with n_workers is set up
                    client=client, n_workers=args.n_workers)
    runtime, history, pareto_pop, pareto_fit = modehb.run(total_cost=args.runtime, verbose=args.verbose,
                                                          # arguments below are part of **kwargs shared across workers
                                                          single_node_with_gpus=single_node_with_gpus, device=device)
    # end of DEHB optimisation

    # Saving optimisation trace history
    name = time.strftime("%x %X %Z", time.localtime(modehb.start))
    name = name.replace("/", '-').replace(":", '-').replace(" ", '_')
    modehb.logger.info("Saving optimisation trace history...")
    with open(os.path.join(args.output_path, "history_{}.pkl".format(name)), "wb") as f:
        pickle.dump(history, f)
    modehb.logger.info("pareto population:{}",pareto_pop)
    modehb.logger.info("pareto fitness:{}",pareto_fit)
    modehb.logger.debug("runtime:{}",runtime)



if __name__ == "__main__":
    main()
