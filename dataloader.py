from mxnet.gluon import data
from DC import DC
from sampler import get_sampler


def dataloader(args):
    if args.only_test is False:
        train_dataset = DC(args, mode='train')
        sampler = get_sampler(args=args, dataset=train_dataset)
        train_loader = data.DataLoader(dataset=train_dataset,
                                       sampler=sampler,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers)
    else:
        train_loader = None
    test_dataset = DC(args, mode='test')
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    return train_loader, test_loader

