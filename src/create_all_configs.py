import argparse

from create_configs import main as default_main
from create_configs_multi import main as multi_main



def main(args):
    default_main(args)

    args.opt = ['AdamW', 'SGD']
    args.head = ['CosFace', 'ArcFace', 'MagFace']
    multi_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


    parser = argparse.ArgumentParser(description='Timm CelebA Training')

    # User config
    parser.add_argument("--user_config", type=str)

    # Model parameters
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--pretrained', default=False)
    parser.add_argument(
        '--head',
        default='CosFace',
        type=str)
    parser.add_argument('--train_loss', default='Focal', type=str)
    parser.add_argument('--min_num_images', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--groups_to_modify',
                        default=['male', 'female'],
                        type=str,
                        nargs='+')
    parser.add_argument('--p_identities',
                        default=[1.0, 1.0],
                        type=float,
                        nargs='+')
    parser.add_argument('--p_images',
                        default=[1.0, 1.0],
                        type=float,
                        nargs='+')
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--name', default='vggface2', type=str)
    parser.add_argument('--dataset', default='vggface2', type=str)
    parser.add_argument('--file_name',
                        default='timm_from-scratch.csv',
                        type=str)
    parser.add_argument('--file_name_ema',
                        default='timm_from-scratch_ema.csv',
                        type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--out_dir', default=".", type=str)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

    # Optimizer parameters
    parser.add_argument('--opt',
                        default="Adam",
                        type=str)
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

    # Regularization parameters

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

    # Model Exponential Moving Average
    parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

    args = parser.parse_args()
    main(args)


