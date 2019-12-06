import argparse


def str2bool(v):
    return v.lower() in ('true')


parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_models_dir', default='D:/weights_results/Dogs_vs_Cats_MXNET/pretrained_models')
parser.add_argument('--use_network', type=str, default='RES', help="\'VGG\' or \'RES\'")
parser.add_argument('--vgg_type', type=str, default='vgg16_bn',
                    help="\'vgg16\',\'vgg19\', \'vgg16_bn\' or \'vgg19_bn\'")

parser.add_argument('--use_tensorboard', type=str2bool, default=True)
parser.add_argument('--use_id_loss', type=str2bool, default=True)

parser.add_argument('--use_gpu', type=str2bool, default=True)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--batch_type', type=str, default='random',
                    help="sampler type option: \'random\' or \'seq\'")

parser.add_argument('--need_test', type=str2bool, default=True)
parser.add_argument('--only_test', type=str2bool, default=True)
parser.add_argument('--resume', type=str2bool, default=False)
parser.add_argument('--epoch_test', type=str2bool, default=False)
parser.add_argument('--checkpoint_epoch', type=int, default=0)

parser.add_argument('--optim_phase', type=str, default='MultiFactor',
                    help="learning rate scheduler option:\'Factor\', \'MultiFactor\', \'Poly\' or \'Cosine\'")
parser.add_argument('--lr_decay_epochs', type=list, default=[1, 3, 5], help="For MultiFactorScheduler step")
parser.add_argument('--every_lr_decay_step', type=int, default=2, help="For FactorScheduler step")
parser.add_argument('--epochs', type=int, default=10)

parser.add_argument('--optimizer', type=str, default='Adam', help="\'Adam\' or \'SGD\'")
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--epsilon', type=float, default=2e-8)
parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--train_dir', type=str, default="D:/datasets/dogs-vs-cats/train")
parser.add_argument('--test_dir', type=str, default="D:/datasets/dogs-vs-cats/test")

parser.add_argument('--experiment_dir', type=str, default="D:/weights_results/Dogs_vs_Cats_MXNET")
parser.add_argument('--epochs_per_val', type=int, default=2)
parser.add_argument('--steps_per_log', type=int, default=10)

parser.add_argument('--transform_type', type=str, default="official", help="\'official\' or \'self_designed\'")
parser.add_argument('--sumbol_or_ndarray', type=str, default='ndarray',
                    help='self_designed_transform function need a data_type as input.')
parser.add_argument('--CenterCrop', type=str2bool, default=True)
parser.add_argument('--CenterCropSize', type=tuple, default=(224, 224))
parser.add_argument('--Normalize', type=str2bool, default=True)
parser.add_argument('--mean', default=(0.485, 0.456, 0.406))
parser.add_argument('--std', default=(0.229, 0.224, 0.225))
args, _ = parser.parse_known_args()
