# helper functions for training
import os, sys
import shutil

import torch
from torch.nn import init


def reset_learning_rate(optimizer, args):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


def adjust_learning_rate(optimizer, epoch, args):
    # old_lr = optimizer.param_groups[0]['lr']
    if args.custom_lr:
        # lr = args.lr
        # try:
        #import ipdb; ipdb.set_trace()
        pointer = next(x[0] for x in enumerate(args.lr_switch_epochs) if epoch >= x[1])
        lr = args.lrs[pointer]
        # except StopIteration:
        #     pass
    else:
        lr = args.lr * (args.lr_decay_rate ** (epoch // args.lr_decay_epochs))
        lr = max(lr, args.lr_clip)
    # logger.log('lr: ' + str(lr))

    #reset_learning_rate(optimizer, args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_weights_multi(m, init_type, gain=1.):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


# ---------------- Pretty sure the following functions/classes are common ----------------
def save_checkpoint(state, is_best, ckpt_dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(ckpt_dir, filename))
    if state['epoch'] % 10 == 1:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'checkpoint_'+str(state['epoch'])+'.pth.tar'))

    if is_best:
        shutil.copyfile(
            os.path.join(ckpt_dir, filename),
            os.path.join(ckpt_dir, 'model_best.pth.tar'))


class Logger(object):
    def __init__(self, out_fname):
        self.out_fd = open(out_fname, 'w')

    def log(self, out_str, end='\n'):
        """
        out_str: single object now
        """
        self.out_fd.write(str(out_str) + end)
        self.out_fd.flush()
        print(out_str, end=end, flush=True)

    def close(self):
        self.out_fd.close()


class MovingAverage(object):
    def __init__(self, N):
        self.cumsum = [0]
        self.moving_avgs = []
        self.N = N
        self.counter = 1

    def update(self, x):
        self.cumsum.append(self.cumsum[self.counter - 1] + x)

        if self.counter < self.N:
            self.moving_avgs.append(self.cumsum[self.counter] / self.counter)
        else:
            moving_avg = (self.cumsum[self.counter] - self.cumsum[self.counter - self.N]) / self.N
            self.moving_avgs.append(moving_avg)
        self.counter += 1
        return self.moving_avgs[-1]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
