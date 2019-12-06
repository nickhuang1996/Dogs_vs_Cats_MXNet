from collections import OrderedDict
from Loss.ID_loss import IDLoss


def set_loss(args, tb_writer=None):
    loss_functions = OrderedDict()
    if args.use_id_loss is True:
        loss_functions['IDLoss'] = IDLoss(tb_writer=tb_writer)
    else:
        raise NotImplementedError
    return loss_functions
