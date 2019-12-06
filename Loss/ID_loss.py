from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet import nd

class IDLoss(object):
    def __init__(self, tb_writer=None):
        super(IDLoss, self).__init__()
        self.criterion = SoftmaxCrossEntropyLoss()
        self.name = 'IDLoss'
        self.tb_writer = tb_writer

    def __call__(self, outputs, labels, train_total=0):
        loss = self.criterion(outputs, labels).mean()
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag=self.name,
                                       tag_scalar_dict={
                                          self.name: loss.astype('float32')
                                       },
                                       global_step=train_total,
                                       )
        return loss

