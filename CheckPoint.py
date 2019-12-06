import os
import os.path as osp

class CheckPoint(object):
    def __init__(self, args, trainer_log, ctx):
        super(CheckPoint, self).__init__()
        self.experiment_dir = args.experiment_dir
        self.use_network = args.use_network
        self.checkpoint_dir = self.experiment_dir + '/checkpoints/' + self.use_network
        if not osp.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            print("The checkpoint dir has been created:{}".format(self.checkpoint_dir))
        self.trainer_log = trainer_log
        self.ctx = ctx

    def save_checkpoint_parameters(self, epoch, model, current_accuracy, best_accuracy):
        save_path = self.checkpoint_dir + '/' + str(epoch) + '.params'
        model.save_parameters(save_path)
        self.trainer_log.print_epoch_log(epoch=epoch)
        if current_accuracy >= best_accuracy:
            best_accuracy = current_accuracy
            best_save_path = self.checkpoint_dir + '/best.params'
            model.save_parameters(best_save_path)
            self.trainer_log.print_best_accuracy_log(epoch=epoch, best_accuracy=best_accuracy)
        return best_accuracy

    def load_checkpoint_parameters(self, epoch, model, epoch_test=False):
        if epoch_test is True:
            load_path = self.checkpoint_dir + '/' + str(epoch) + '.params'
        else:
            load_path = self.checkpoint_dir + '/best.params'
        assert os.path.exists(load_path), "{} is not found!".format(load_path)
        model.load_parameters(load_path, ctx=self.ctx)
        resume_epoch = epoch + 1
        self.trainer_log.print_load_checkpoint_log(load_path=load_path)
        return resume_epoch
