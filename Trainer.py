from dataloader import dataloader
from Networks import get_networks
from device import set_ctx
from optimizer import set_optimizer
from Loss import set_loss
from TrainerLog import TrainerLog
from CheckPoint import CheckPoint
from lr_scheduler import get_lr_scheduler
from mxnet import autograd
from mxnet import nd
from mxnet import metric

import os.path as osp
import os


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.experiment_dir = args.experiment_dir
        if not osp.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            print("The experiment dir has been created:{}".format(self.experiment_dir))
        self.trainer_log = TrainerLog(args=args, append=True)
        self.ctx = set_ctx(args=args)
        self.check_point = CheckPoint(args=args, trainer_log=self.trainer_log, ctx=self.ctx)
        self.train_loader, self.test_loader = dataloader(args=args)
        self.lr_scheduler = None
        self.optimizer = None
        self.model = None
        if self.train_loader is not None:
            self.train_samples_num = self.train_loader._dataset.__len__()
            print("train dataset samples: {}".format(self.train_samples_num))
        self.test_samples_num = self.test_loader._dataset.__len__()
        print("test dataset samples: {}".format(self.test_samples_num))
        self.resume_epoch = 0
        if args.only_test is False:
            if args.use_tensorboard is True:
                from tensorboardX import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=osp.join(args.experiment_dir, 'tensorboard'))
            else:
                self.tb_writer = None
            if args.resume is True:
                self.checkpoint_epoch = args.checkpoint_epoch
                self.model = get_networks(args=args, ctx=self.ctx)
                self.resume_epoch = self.check_point.load_checkpoint_parameters(epoch=self.checkpoint_epoch,
                                                                                model=self.model)
            else:
                self.model = get_networks(args=args, ctx=self.ctx)
                self.model.classifier.initialize(ctx=self.ctx)

            self.lr_scheduler = get_lr_scheduler(args=args, train_loader=self.train_loader)
            self.optimizer, self.trainer = set_optimizer(model=self.model, lr_scheduler=self.lr_scheduler, args=args)
            self.loss_functions = set_loss(args=args, tb_writer=self.tb_writer)
            self.current_epoch = None
        elif args.only_test is True:
            self.checkpoint_epoch = args.checkpoint_epoch
            self.model = get_networks(args=args, ctx=self.ctx)
            self.epoch_test = args.epoch_test
            _ = self.check_point.load_checkpoint_parameters(epoch=self.checkpoint_epoch,
                                                            model=self.model,
                                                            epoch_test=self.epoch_test)
        if self.lr_scheduler is not None:
            self.trainer_log.print_use_lr_scheduler()
        if self.optimizer is not None and self.trainer is not None:
            self.trainer_log.print_use_optimizer()
        if self.model is not None:
            self.trainer_log.print_use_network()
        self.test_accuracy_metric = metric.Accuracy()
        self.epochs = args.epochs
        self.train_total = 0
        self.best_accuracy = None
        self.current_accuracy = None

    def train(self, need_test=False):
        print("Training process starts from epoch {}...".format(self.resume_epoch))
        for epoch in range(self.resume_epoch, self.epochs):
            self.current_epoch = epoch
            for _, item in enumerate(self.train_loader):
                inputs, labels = item
                inputs = inputs.as_in_context(self.ctx)
                labels = labels.as_in_context(self.ctx)
                cls = 0.0
                with autograd.record():  # Gradient
                    outputs = self.model(inputs)
                    for _, loss_type in self.loss_functions.items():
                        cls += loss_type(outputs=outputs, labels=labels, train_total=self.train_total)
                cls.backward()

                self.train_total += inputs.shape[0]
                self.trainer.step(batch_size=inputs.shape[0])

                cls = nd.array(cls).asscalar()
                if self.train_total % self.args.steps_per_log == 0:
                    self.trainer_log.print_batch_log(current_lr=self.lr_scheduler.base_lr,
                                                     current_epoch=self.current_epoch,
                                                     epochs=self.epochs,
                                                     train_total=self.train_total,
                                                     loss=cls,
                                                     )

            nd.waitall()
            if (epoch + 1) % self.args.epochs_per_val == 0:
                if need_test is True:
                    self.test()
                self.best_accuracy = self.check_point.save_checkpoint_parameters(epoch=self.current_epoch,
                                                                                 model=self.model,
                                                                                 current_accuracy=self.current_accuracy,
                                                                                 best_accuracy=self.best_accuracy)
        self.trainer_log.log_close()

    def test(self):
        from tqdm import tqdm
        if self.args.only_test is True:
            epoch = self.checkpoint_epoch
        else:
            epoch = self.current_epoch
        test_accuracy = 0.0
        for item in tqdm(self.test_loader, desc='Extract Feature', miniters=20, ncols=120, unit=' batches'):
            inputs, labels = item
            inputs = inputs.as_in_context(self.ctx)
            labels = labels.as_in_context(self.ctx)
            outputs = self.model(inputs)
            preds = nd.argmax(outputs, axis=1)
            self.test_accuracy_metric.update(preds=preds, labels=labels)
            test_accuracy = self.test_accuracy_metric.get()[1]

        if self.best_accuracy is None:
            self.best_accuracy = test_accuracy
        self.current_accuracy = test_accuracy
        self.trainer_log.print_test_accuracy_log(epoch=epoch,
                                                 epochs=self.epochs,
                                                 current_accuracy=self.current_accuracy,
                                                 best_accuracy=self.best_accuracy)
        self.test_accuracy_metric.reset()
        if self.args.only_test is True:
            self.trainer_log.log_close()



