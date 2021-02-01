import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from backbone import *     # 会调用backbone中的__init__，继而注册build_mobilefacenet
from kdreid import *


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_shufflenet_config(cfg)    # 先add_config, 再merge修改
    add_mobilefacenet_config(cfg)
    add_kdreid_config(cfg)
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)  # Set up the detectron2 logger, Log basic info, Backup the config
    return cfg


def get_parser():
    """ Returns: argparse.ArgumentParser """
    parser = argparse.ArgumentParser(description="my fastreid training")

    parser.add_argument("--config-file", 
        default="./my_project/configs/Base-bagtricks.yml", 
        metavar="FILE", 
        help="path to config file"
    )
    parser.add_argument("--kd", 
        default=True,      
        # action="store_true", 
        help="kd training with teacher model guided")
    parser.add_argument("--finetune",
        # action="store_true",
        default=False,
        help="whether to attempt to finetune from the trained model",
    )
    parser.add_argument("--resume",
        # action="store_true",
        default=False,
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", 
        # action="store_true",
        default=False,
        help="perform evaluation only"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus per machine")

    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    # parser.add_argument("--dist-url", default="env://")

    parser.add_argument("--opts",
        help="Modify config options using the command-line",
        # default=None,
        default=[
            # 'MODEL.DEVICE', 'cuda:1',
            # 'MODEL.STUDENT_WEIGHTS', '/data_4t/xk/ReID/fast-reid-master/logs/market1501/12_bagtricks_mobileFN_GDConv_SGD_lr0p01_bnprelu_kd/model_final.pth',
            'MODEL.TEACHER_WEIGHTS', '/data_4t/xk/ReID/fast-reid-master/logs/market1501/1_bagtricks_R50_20201125_Adam_lr0p00035/model_final.pth'
                ],
        nargs=argparse.REMAINDER, 
    )
    return parser


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = DefaultTrainer.test(cfg, model)
        return res

    if args.kd: 
        trainer = KDTrainer(cfg)
    else:       
        trainer = DefaultTrainer(cfg)

    # trainer = DefaultTrainer(cfg)
    # if args.finetune: 
    #     Checkpointer(trainer.model).load(cfg.MODEL.WEIGHTS)  # load trained model to funetune

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
