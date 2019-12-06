from .ResidualNet import ResidualNet
from .VGGNet import VGGNet

networks_factory = {
    'RES': ResidualNet,
    'VGG': VGGNet,
}


def get_networks(args, ctx=None):
    return networks_factory[args.use_network](args=args, ctx=ctx)
