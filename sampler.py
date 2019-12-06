from mxnet.gluon.data.sampler import RandomSampler
from mxnet.gluon.data.sampler import SequentialSampler


def get_sampler(args, dataset):
    if args.batch_type == 'seq':
        sampler = SequentialSampler(len(dataset))
    elif args.batch_type == 'random':
        sampler = RandomSampler(len(dataset))
    else:
        raise NotImplementedError
    return sampler
