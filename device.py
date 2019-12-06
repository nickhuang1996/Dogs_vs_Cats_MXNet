import mxnet


def set_ctx(args):
    use_gpu = args.use_gpu
    ctx = mxnet.gpu() if use_gpu else mxnet.cpu()
    return ctx
