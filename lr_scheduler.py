from bisect import bisect_right

import torch
from torch.optim import Optimizer
from math import pi, cos
import warnings


def flat_and_anneal_lr_scheduler(
    optimizer,
    total_iters,
    warmup_iters=0,
    warmup_factor=0.1,
    warmup_method="linear",
    anneal_point=0.72,
    anneal_method="cosine",
    target_lr_factor=0,
    poly_power=1.0,
    step_gamma=0.1,
    steps=[2 / 3.0, 8 / 9.0],
    return_function=False,
):
    """Ref: https://github.com/fastai/fastai/blob/master/fastai/callbacks/flat_cos_anneal.py.

    warmup_initial_lr = warmup_factor * base_lr
    target_lr = base_lr * target_lr_factor
    total_iters: cycle length; set to max_iter to get a one cycle schedule.
    """
    if warmup_method not in ("constant", "linear"):
        raise ValueError("Only 'constant' or 'linear' warmup_method accepted," "got {}".format(warmup_method))

    if anneal_method not in ("cosine", "linear", "poly", "exp", "step", "none"):
        raise ValueError(
            "Only 'cosine', 'linear', 'poly', 'exp', 'step' or 'none' anneal_method accepted,"
            "got {}".format(anneal_method)
        )

    if anneal_method == "step":
        if any([_step < warmup_iters / total_iters or _step > 1 for _step in steps]):
            raise ValueError(
                "error in steps: {}. warmup_iters: {} total_iters: {}."
                "steps should be in ({},1)".format(steps, warmup_iters, total_iters, warmup_iters / total_iters)
            )
        if list(steps) != sorted(steps):
            raise ValueError("steps {} is not in ascending order.".format(steps))
        warnings.warn("ignore anneal_point when using step anneal_method")
        anneal_start = steps[0] * total_iters
    else:
        if anneal_point > 1 or anneal_point < 0:
            raise ValueError("anneal_point should be in [0,1], got {}".format(anneal_point))
        anneal_start = anneal_point * total_iters

    def f(x):  # x is the iter in lr scheduler, return the lr_factor
        # the final lr is warmup_factor * base_lr
        x = x % total_iters  # cyclic
        if x < warmup_iters:
            if warmup_method == "linear":
                alpha = float(x) / warmup_iters
                return warmup_factor * (1 - alpha) + alpha
            elif warmup_method == "constant":
                return warmup_factor
        elif x >= anneal_start:
            if anneal_method == "step":
                # ignore anneal_point and target_lr_factor
                milestones = [_step * total_iters for _step in steps]
                lr_factor = step_gamma ** bisect_right(milestones, float(x))
            elif anneal_method == "cosine":
                # slow --> fast --> slow
                lr_factor = target_lr_factor + 0.5 * (1 - target_lr_factor) * (
                    1 + cos(pi * ((float(x) - anneal_start) / (total_iters - anneal_start)))
                )
            elif anneal_method == "linear":
                # (y-m) / (B-x) = (1-m) / (B-A)
                lr_factor = target_lr_factor + (1 - target_lr_factor) * (total_iters - float(x)) / (
                    total_iters - anneal_start
                )
            elif anneal_method == "poly":
                # slow --> fast if poly_power < 1
                # fast --> slow if poly_power > 1
                # when poly_power == 1.0, it is the same with linear
                lr_factor = (
                    target_lr_factor
                    + (1 - target_lr_factor) * ((total_iters - float(x)) / (total_iters - anneal_start)) ** poly_power
                )
            elif anneal_method == "exp":
                # fast --> slow
                # do not decay too much, especially if lr_end == 0, lr will be
                # 0 at anneal iter, so we should avoid that
                _target_lr_factor = max(target_lr_factor, 5e-3)
                lr_factor = _target_lr_factor ** ((float(x) - anneal_start) / (total_iters - anneal_start))
            else:
                lr_factor = 1
            return lr_factor
        else:  # warmup_iter <= x < anneal_start_iter
            return 1

    if return_function:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f), f
    else:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def test_flat_and_anneal():
    from mmcv import Config
    import numpy as np

    model = resnet18()
    base_lr = 1e-4
    optimizer_cfg = dict(type="Adam", lr=base_lr, weight_decay=0)
    optimizer = obj_from_dict(optimizer_cfg, torch.optim, dict(params=model.parameters()))

    # learning policy
    total_epochs = 80
    epoch_len = 500
    total_iters = epoch_len * total_epochs // 2
    # poly, step, linear, exp, cosine
    lr_cfg = Config(
        dict(
            # anneal_method="cosine",
            # anneal_method="linear",
            # anneal_method="poly",
            # anneal_method="exp",
            anneal_method="step",
            warmup_method="linear",
            step_gamma=0.1,
            warmup_factor=0.1,
            warmup_iters=800,
            poly_power=5,
            target_lr_factor=0.0,
            steps=[0.5, 0.75, 0.9],
            anneal_point=0.72,
        )
    )

    # scheduler = build_scheduler(lr_config, optimizer, epoch_length)
    scheduler = flat_and_anneal_lr_scheduler(
        optimizer=optimizer,
        total_iters=total_iters,
        warmup_method=lr_cfg.warmup_method,
        warmup_factor=lr_cfg.warmup_factor,
        warmup_iters=lr_cfg.warmup_iters,
        anneal_method=lr_cfg.anneal_method,
        anneal_point=lr_cfg.anneal_point,
        target_lr_factor=lr_cfg.target_lr_factor,
        poly_power=lr_cfg.poly_power,
        step_gamma=lr_cfg.step_gamma,
        steps=lr_cfg.steps,
    )
    print("start lr: {}".format(scheduler.get_lr()))
    steps = []
    lrs = []

    epoch_lrs = []
    global_step = 0

    start_epoch = 0
    for epoch in range(start_epoch):
        for batch in range(epoch_len):
            scheduler.step()  # when no state_dict availble
            global_step += 1

    for epoch in range(start_epoch, total_epochs):
        # if global_step >= lr_config['warmup_iters']:
        #     scheduler.step(epoch)
        # print(type(scheduler.get_lr()[0]))
        # import pdb;pdb.set_trace()
        epoch_lrs.append([epoch, scheduler.get_lr()[0]])  # only get the first lr (maybe a group of lrs)
        for batch in range(epoch_len):
            # if global_step < lr_config['warmup_iters']:
            #     scheduler.step(global_step)
            cur_lr = scheduler.get_lr()[0]
            if global_step == 0 or (len(lrs) >= 1 and cur_lr != lrs[-1]):
                print("epoch {}, batch: {}, global_step:{} lr: {}".format(epoch, batch, global_step, cur_lr))
            steps.append(global_step)
            lrs.append(cur_lr)
            global_step += 1
            scheduler.step()  # usually after optimizer.step()
    # print(epoch_lrs)
    # import pdb;pdb.set_trace()
    # epoch_lrs.append([total_epochs, scheduler.get_lr()[0]])

    epoch_lrs = np.asarray(epoch_lrs, dtype=np.float32)
    for i in range(len(epoch_lrs)):
        print("{:02d} {}".format(int(epoch_lrs[i][0]), epoch_lrs[i][1]))

    plt.figure(dpi=100)
    plt.suptitle("{}".format(dict(lr_cfg)), size=4)
    plt.subplot(1, 2, 1)
    plt.plot(steps, lrs, "-.")
    # plt.show()
    plt.subplot(1, 2, 2)
    # print(epoch_lrs.dtype)
    plt.plot(epoch_lrs[:, 0], epoch_lrs[:, 1], "-.")
    plt.show()


if __name__ == "__main__":
    from mmcv.runner import obj_from_dict
    import os.path as osp
    from torchvision.models import resnet18
    import matplotlib.pyplot as plt

    cur_dir = osp.dirname(osp.abspath(__file__))

    test_flat_and_anneal()
