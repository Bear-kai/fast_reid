# encoding: utf-8
"""
@author:  kai xiong
@contact: bearkai1992@qq.com
"""

import numpy as np
import torch
import argparse
import sys
sys.path.append('.')
from dedup_utils import setup_cfg, Robot_camera, Warp_detector, Warp_predictor, get_box_num, Gallery

from torch.backends import cudnn
cudnn.benchmark = True

gallery = {}     # 行人ID特征库
# delta = 0        # 自开机或上一次聚类开始，新增的行人patch数


def get_parser():

    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(        ##
        "--config-file",
        metavar="FILE",
        default="/data_4t/xk/ReID/fast-reid-master/configs/Market1501/mgn_R50-ibn.yml",
        # default="/data_4t/xk/ReID/fast-reid-master/configs/Market1501/AGW_S50.yml", 
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        # action='store_true',
        default=False,
        help='if use multiprocess for feature extraction.'
    )
    
    parser.add_argument(
        "--output",
        default="./vis_rank_list/mgn_r50",
        help="a file or directory to save rankling list result.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', '/data_4t/xk/ReID/fast-reid-master/model_zoo/market_mgn_R50-ibn.pth',
                 'MODEL.DEVICE', 'cuda:1'
                ], 
        # default=['MODEL.WEIGHTS', '/data_4t/xk/ReID/fast-reid-master/model_zoo/market_agw_S50.pth'],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--video-dir",
        # default= '/data_4t/xk/datasets/ReID/robot/night8_speed0.0',
        # default= '/data_4t/xk/datasets/ReID/robot/night8_speed0.5',
        default= '/data_4t/xk/datasets/ReID/robot/night8_speed1.0',
    )

    return parser


def main():
    # config
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    # camera, reid, detector, gallery
    CAMS = Robot_camera(args.video_dir, save_name='demo_speed10_T94_addlogic')
    Reid_gallery = Gallery(thresh_sim=0.94, thresh_stamp=2000)
    PREDICTOR = Warp_predictor(cfg, input_size=(128,384), device='1')
    DETECTOR = Warp_detector(device='0')

    # start demo
    count = 0
    freq = 10                   # 各路相机均每freq秒处理一帧
    new_num, box_num, box_num_0 = 0, 0, 0     # 占位
    count_ls = []
    box_num_ls = []
    allert_ind_ls = []

    while True:
        # read camera
        good_ls, frame_ls = CAMS.read_3_cams()
        count += 1
        if count % freq != 0:
            continue
        if not all(good_ls):
            print("camera over: %d"%count)
            break
        
        # detection and extraction
        boxes_ls = DETECTOR.detect_3_frames(frame_ls)   # list of tensor
        new_num = get_box_num(boxes_ls)
        if new_num:
            patches_ls = PREDICTOR.get_patches_by_cam(boxes_ls, frame_ls)
            feats = PREDICTOR.extract_patches(patches_ls)

            # 初始建库
            if not Reid_gallery.pool_dic:
                Reid_gallery.create(feats, count, patches_ls)
                CAMS.write_4_frames(frame_ls, Reid_gallery.img_dic, count)
                CAMS.allert_reid += 1
                CAMS.allert_box += 1
                box_num = new_num
                box_num_0 = new_num
                continue

            # 匹配更新
            match_dic = Reid_gallery.match_update(feats, count, patches_ls)

            # 删过期ID
            Reid_gallery.check_stamp(count)     # del_key_ls = 
            
            # 报警计数bbox: 检测上升沿
            if new_num > box_num_0:
                CAMS.allert_box_0 += 1
            box_num_0 = new_num

            # 报警计数bbox: 检测上升沿/下降沿并考虑稳定帧数
            if new_num > box_num:
                CAMS.allert_upflag = 1
                CAMS.allert_updwtime = 1
                box_num = new_num
            elif new_num == box_num:
                CAMS.allert_updwtime += 1
                if CAMS.allert_updwtime >= CAMS.allert_updwtime_thresh:
                    if CAMS.allert_upflag:
                        CAMS.allert_upflag = 0
                        CAMS.allert_box += 1
                        allert_ind_ls.append(count)     # temp
            elif new_num < box_num:
                CAMS.allert_upflag = 0
                CAMS.allert_updwtime = 1
                box_num = new_num
            # temp
            count_ls.append(count)
            box_num_ls.append(new_num)

            # 报警计数reid：query是否均匹配到gallery
            if not match_dic['all_match']:
                CAMS.allert_reid += 1

            # 显示库信息 & 写视频
            CAMS.show_info()
            Reid_gallery.show_info(count)
            CAMS.write_4_frames(frame_ls, Reid_gallery.img_dic, count)

    CAMS.videoWriter.release()
    # temp
    assert len(box_num_ls) == len(count_ls)
    with open('./test.txt', 'w') as fp:
        for i, cnt in enumerate(count_ls):
            info = 'frame %d, box_num %d'%(cnt, box_num_ls[i])
            if cnt in allert_ind_ls:
                info += ', allert +1'
            fp.write(info + '\n')

    return 0


if __name__ == '__main__':
    main()