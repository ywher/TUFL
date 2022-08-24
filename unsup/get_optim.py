import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)
from utils.optimizer import Optimizer

def get_optim(args, net):
    momentum = args.momentum
    weight_decay = args.weight_decay
    lr_start = args.lr_start
    max_iter = args.max_iter
    power = args.lr_power
    warmup_steps = int(args.warm_up_ratio*max_iter)
    warmup_start_lr = args.warmup_start_lr
    optim = Optimizer(
            model = net.module,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    return optim