from mxnet.gluon.data.vision.transforms import Compose
from mxnet.gluon.data.vision.transforms import Normalize
from mxnet.gluon.data.vision.transforms import ToTensor
from mxnet.gluon.data.vision.transforms import Resize
from mxnet.gluon.data.vision.transforms import CenterCrop

from mxnet.image import center_crop
import mxnet
two_type_factory = {
    'symbol': mxnet.symbol,
    'ndarray': mxnet.ndarray
}


def self_designed_transform(x, args):
    if args.CenterCrop is True:
        x, _ = center_crop(x, args.CenterCropSize)
    x = two_type_factory[args.sumbol_or_ndarray].image.to_tensor(x)
    if args.Normalize is True:
        x = two_type_factory[args.sumbol_or_ndarray].image.normalize(x, mean=args.mean, std=args.std)
    return x


def official_transform(x, args):
    transformer = Compose([
        CenterCrop(args.CenterCropSize),
        ToTensor(),
        Normalize(mean=args.mean, std=args.std),
    ])
    x = transformer(x)
    return x


transform_factory = {
    'self_designed': self_designed_transform,
    'official': official_transform
}


def transform(x, args):
    return transform_factory[args.transform_type](x, args)
