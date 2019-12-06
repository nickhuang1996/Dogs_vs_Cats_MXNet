from mxnet.lr_scheduler import FactorScheduler
from mxnet.lr_scheduler import MultiFactorScheduler
from mxnet.lr_scheduler import PolyScheduler
from mxnet.lr_scheduler import CosineScheduler


def get_lr_scheduler(args, train_loader):
    if args.optim_phase == 'Factor':
        every_lr_decay_step = args.every_lr_decay_step
        lr_scheduler = FactorScheduler(step=every_lr_decay_step, factor=0.1)
    elif args.optim_phase == 'MultiFactor':
        lr_decay_steps = [len(train_loader) * ep for ep in args.lr_decay_epochs]
        lr_scheduler = MultiFactorScheduler(step=lr_decay_steps, factor=0.1)
    elif args.optim_phase == 'Poly':
        max_update_step = args.epochs
        lr_scheduler = PolyScheduler(max_update=max_update_step)
    elif args.optim_phase == 'Cosine':
        max_update_step = args.epochs
        lr_scheduler = CosineScheduler(max_update=max_update_step)
    else:
        raise ValueError('Invalid phase {}'.format(args.optim_phase))
    return lr_scheduler


