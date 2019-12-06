from mxnet.gluon.model_zoo.vision import resnet50_v2
from mxnet.gluon import nn


class ResidualNet(nn.Block):
    def __init__(self, args, ctx):
        super(ResidualNet, self).__init__()
        self.args = args
        resnet = resnet50_v2(pretrained=True,
                             ctx=ctx,
                             root=args.pretrained_models_dir)
        self.features = resnet.features

        self.classifier = nn.Sequential()
        with self.classifier.name_scope():
            self.classifier.add(
                # nn.Flatten(),
                nn.Dense(2048, in_units=2048, activation='relu'),
                nn.Dropout(0.5),
                nn.Dense(2048, in_units=2048, activation='relu'),
                nn.Dropout(0.5),
                nn.Dense(2, in_units=2048)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



