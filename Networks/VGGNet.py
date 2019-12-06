from mxnet.gluon.model_zoo.vision import vgg16
from mxnet.gluon.model_zoo.vision import vgg19
from mxnet.gluon.model_zoo.vision import vgg16_bn
from mxnet.gluon.model_zoo.vision import vgg19_bn
from mxnet.gluon import nn

vgg_features_factory = {
    'vgg16': vgg16,
    'vgg19': vgg19,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
}


class VGGNet(nn.Block):
    def __init__(self, args, ctx):
        super(VGGNet, self).__init__()
        use_vgg = vgg_features_factory[args.vgg_type](pretrained=True,
                                                      ctx=ctx,
                                                      root=args.pretrained_models_dir)

        self.features = use_vgg.features
        self.classifier = nn.Dense(2,
                                   in_units=4096,
                                   weight_initializer='normal',
                                   bias_initializer='zeros')

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
