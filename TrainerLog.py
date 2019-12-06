import os.path as osp
import os


class TrainerLog(object):
    def __init__(self, args, append=True):
        super(TrainerLog, self).__init__()
        self.args = args
        self.experiment_dir = args.experiment_dir
        self.lr_scheduler = args.optim_phase
        self.optimizer = args.optimizer
        self.use_network = args.use_network
        self.log_dir = self.experiment_dir + '/log'
        if self.args.only_test is True:
            log_dir = osp.join(self.log_dir, self.use_network, 'test', self.time_str())
        else:
            log_dir = osp.join(self.log_dir, self.use_network, 'train', self.time_str())
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
            print("The log dir has been created:{}".format(log_dir))
        accuracy_file_path = osp.join(log_dir, 'accuracy.txt')
        log_file_path = osp.join(log_dir, 'log.txt')
        self.accuracy_file = open(accuracy_file_path, "a" if append else "w")
        self.log_file = open(log_file_path, "a" if append else "w")

    def print_use_lr_scheduler(self):
        print("lr_scheduler: {}".format(self.lr_scheduler))
        self.log_file.write("Optimizer: {}".format(self.lr_scheduler))
        self.log_file.write('\n')
        self.log_file.flush()

    def print_use_optimizer(self):
        print("Optimizer: {}".format(self.optimizer))
        self.log_file.write("Optimizer: {}".format(self.optimizer))
        self.log_file.write('\n')
        self.log_file.flush()

    def print_use_network(self):
        print("Use network is: {}".format(self.use_network))
        self.log_file.write("Use network is: {}".format(self.use_network))
        self.log_file.write('\n')
        self.log_file.flush()

    @staticmethod
    def time_str(fmt=None):
        import datetime
        if fmt is None:
            fmt = '%Y-%m-%d_%H-%M-%S'
        return datetime.datetime.today().strftime(fmt)

    def log_close(self):
        self.accuracy_file.close()
        self.log_file.close()

    def print_batch_log(self, current_lr, current_epoch, epochs, train_total, loss):
        learning_rate = '{:.7f}'.format(current_lr).rstrip('0')
        print("Epoch %d/%d learning_rate " % (current_epoch, epochs)
              + learning_rate
              + " train_total %5d train_loss %.4f" % (train_total, loss))
        self.log_file.write("Epoch %d/%d learning_rate " % (current_epoch, epochs)
                            + learning_rate
                            + " train_total %5d train_loss %.4f" % (train_total, loss))
        self.log_file.write('\n')
        self.log_file.flush()

    def print_epoch_log(self, epoch):
        print("Checkpoint at epoch {} has been saved.".format(epoch))
        self.log_file.write("Checkpoint at epoch {} has been saved.".format(epoch))
        self.log_file.write('\n')
        self.log_file.flush()

    def print_load_checkpoint_log(self, load_path):
        print("Load checkpoint file {}".format(load_path))
        self.log_file.write("Load checkpoint file {}".format(load_path))
        self.log_file.write('\n')
        self.log_file.flush()

    def print_test_accuracy_log(self, epoch, epochs, current_accuracy, best_accuracy):
        if self.args.only_test is False:
            print("Epoch %d/%d test_accuracy %.2f%% best_accuracy %.2f%%" % (
                epoch, epochs, current_accuracy * 100, best_accuracy * 100))
            self.log_file.write("Epoch %d/%d test_accuracy %.2f%% best_accuracy %.2f%%" % (
                epoch, epochs, current_accuracy * 100, best_accuracy * 100))
            self.log_file.write('\n')
            self.log_file.flush()

            self.accuracy_file.write("epoch: %d test_accuracy:%.2f%%" % (epoch, current_accuracy * 100))
            self.accuracy_file.write('\n')
            self.accuracy_file.flush()
        else:
            print("test_accuracy %.2f%%" % (current_accuracy * 100))
            self.log_file.write("test_accuracy %.2f%%" % (current_accuracy * 100))
            self.log_file.write('\n')
            self.log_file.flush()

            self.accuracy_file.write("test_accuracy:%.2f%%" % (current_accuracy * 100))
            self.accuracy_file.write('\n')
            self.accuracy_file.flush()

    def print_best_accuracy_log(self, epoch, best_accuracy):
        print("Best checkpoint at epoch %d has been saved. accuracy is %.2f%%" % (epoch, best_accuracy * 100))
        self.log_file.write("Best checkpoint at epoch %d has been saved. accuracy is %.2f%%" % (epoch, best_accuracy * 100))
        self.log_file.write('\n')
        self.log_file.flush()

        self.accuracy_file.write("epoch: %d best_accuracy:%.2f%%" % (epoch, best_accuracy * 100))
        self.accuracy_file.write('\n')
        self.accuracy_file.flush()
