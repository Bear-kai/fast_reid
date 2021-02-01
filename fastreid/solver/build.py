# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn as nn
from . import lr_scheduler
from . import optim

# ========== modified
# def build_optimizer(cfg, model):
#     """ 2020.12.7 自定义optimizer, 分离BN和PReLU参数, 不加decay """
#     lr = cfg.SOLVER.BASE_LR
#     weight_decay = cfg.SOLVER.WEIGHT_DECAY

#     bn_prelu_params = []
#     ignored_params = []
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             ignored_params += list(map(id, m.parameters()))    # id() 函数用于获取对象的内存地址
#             bn_prelu_params += m.parameters()
#         if isinstance(m, nn.BatchNorm1d):
#             ignored_params += list(map(id, m.parameters()))    
#             bn_prelu_params += m.parameters()
#         elif isinstance(m, nn.PReLU):
#             ignored_params += list(map(id, m.parameters()))
#             bn_prelu_params += m.parameters()
#     base_params = list(filter(lambda p: id(p) not in ignored_params, model.parameters()))

#     params = [ 
#         {'params': base_params,     'weight_decay': weight_decay,  "lr": lr,  "freeze": False},
#         {'params': bn_prelu_params, 'weight_decay': 0.,            "lr": lr,  "freeze": False}
#     ]

#     solver_opt = cfg.SOLVER.OPT
#     # fmt: off
#     if solver_opt == "SGD": 
#         opt_fns = getattr(optim, solver_opt)(params, momentum=cfg.SOLVER.MOMENTUM)
#     else:                   
#         opt_fns = getattr(optim, solver_opt)(params)
#     # fmt: on
#     return opt_fns


# ========== original:
# 不够普适，无法找出BN/PReLU参数，因为对于含sequential构成的block的网络，bn和prelu字样被数字编号代替了
def build_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad: 
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "heads" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]

    solver_opt = cfg.SOLVER.OPT
    # fmt: off
    if solver_opt == "SGD": 
        opt_fns = getattr(optim, solver_opt)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:                   
        opt_fns = getattr(optim, solver_opt)(params)
    # fmt: on
    return opt_fns
    

# ========== 
def build_lr_scheduler(cfg, optimizer):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
        "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        "warmup_method": cfg.SOLVER.WARMUP_METHOD,

        # multi-step lr scheduler options
        "milestones": cfg.SOLVER.STEPS,
        "gamma": cfg.SOLVER.GAMMA,

        # cosine annealing lr scheduler options
        "max_iters": cfg.SOLVER.MAX_ITER,
        "delay_iters": cfg.SOLVER.DELAY_ITERS,
        "eta_min_lr": cfg.SOLVER.ETA_MIN_LR,

    }
    return getattr(lr_scheduler, cfg.SOLVER.SCHED)(**scheduler_args)
