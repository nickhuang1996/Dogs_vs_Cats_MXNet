from parse_setting import args
from Trainer import Trainer

if __name__ == '__main__':
    DCTrainer = Trainer(args=args)
    if args.only_test is False:
        if args.need_test:
            DCTrainer.train(need_test=True)
        else:
            DCTrainer.train(need_test=False)
    else:
        DCTrainer.test()
