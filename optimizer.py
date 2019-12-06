from mxnet.gluon import Trainer
import mxnet


def set_optimizer(model, lr_scheduler, args):
    if args.optimizer == 'Adam':
        kwargs = dict(learning_rate=args.lr,
                      beta1=args.beta1,
                      beta2=args.beta2,
                      epsilon=args.epsilon,
                      lr_scheduler=lr_scheduler)
    elif args.optimizer == 'SGD':
        kwargs = dict(learning_rate=args.lr,
                      momentum=args.momentum,
                      lr_scheduler=lr_scheduler)
    else:
        raise ValueError('Invalid optimizer {}'.format(args.optimizer))
    optimizer = mxnet.optimizer.create(args.optimizer,
                                       **kwargs)
    trainer = Trainer(model.collect_params(), optimizer=optimizer)
    return optimizer, trainer
