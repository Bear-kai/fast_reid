import numpy as np
import torch
import random
import cv2
import os
import io
from PIL import Image
import sys
import logging
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fastreid.engine import DefaultPredictor
from fastreid.config import get_cfg
from yolov5_body_mbv3xyolo_master.detect_body import YoloHand


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_box_num(boxes_ls):
    num = 0
    for boxes in boxes_ls:
        num += boxes.shape[0]

    return num


class Robot_camera():

    def __init__(self, video_dir, save_name='demo', FPS=3):
        """ video_dir含有3路视频，命名:{}-1.mp4, {}-2.mp4, {}-3.mp4 """
        self.cam_dic = {}
        for i, vd_name in enumerate(os.listdir(video_dir)):
            if '.avi' in vd_name:
                continue
            print('read video: %s'%vd_name)
            video_path = os.path.join(video_dir, vd_name)
            camera = cv2.VideoCapture(video_path) 
            ind = vd_name.split('-')[-1].split('.')[0]
            assert ind in ['1','2','3'], 'video should be name as \{\}-1.mp4, \{\}-2.mp4, \{\}-3.mp4'
            self.cam_dic[ind] = camera

        self.WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # FPS = 3                                   # int(camera.get(cv2.CAP_PROP_FPS))
        FOURCC = cv2.VideoWriter_fourcc(*'XVID')    # int(cap.get(cv2.CAP_PROP_FOURCC))
        save_path = os.path.join(video_dir, '%s.avi'%save_name)

        scale = 0.5
        ssize = (self.WIDTH, self.HEIGHT)
        self.margin = 5
        self.dsize = (int(ssize[0]*scale), int(ssize[1]*scale))
        self.whole_size = (self.dsize[0]*2+self.margin, self.dsize[1]*2+self.margin)    # W*H
        self.save_dir = os.path.join(video_dir, 'IDimgs')
        self.videoWriter = cv2.VideoWriter(save_path, FOURCC, FPS, self.whole_size)

        # 报警设置
        self.allert_reid = 0                # 基于reid的报警次数
        self.allert_box_0 = 0               # 基于bbox的报警次数
        self.allert_box = 0                 # 基于bbox并考虑连续帧的报警次数
        self.allert_upflag = 0              # 是否检测到上升沿
        # self.allert_dwflag = 0              # 是否检测到下降沿
        self.allert_updwtime = 0            # 上升/下降沿后的稳定时长/帧长
        self.allert_updwtime_thresh = 2     # 稳定时长/帧长阈值

        self.show_patches = []
        self.patch_size = (128,256)
        self.grid_size = [2*7, 4*14, 8*28, 16*56]  # 128*256, 64*128, 32*64, 16*32

    def read_3_cams(self):
        good_ls = []
        img_ls = []
        for i in range(3):
            good, img = self.cam_dic[str(i+1)].read()
            good_ls.append(good)
            img_ls.append(img)

        return good_ls, img_ls

    def combine_3_frames(self, frame_ls, stamp):
        frame1_s = cv2.resize(frame_ls[0], self.dsize)
        frame2_s = cv2.resize(frame_ls[1], self.dsize)
        frame3_s = cv2.resize(frame_ls[2], self.dsize)

        # show title        
        cv2.putText(frame1_s, 'Frame %d'%stamp, (30, 65), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame1_s, 'box0: box1: reid = %d : %d : %d'%(self.allert_box_0, self.allert_box, self.allert_reid), 
                    (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        for txt, frame in zip(['cam_1','cam_2','cam_3'], [frame1_s,frame2_s,frame3_s]):
            # cv2.rectangle(frame, (0, 0), (len(txt)*20, 35), (0,0,255), cv2.FILLED)
            cv2.putText(frame, txt, (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  # not support Chinese

        arr = np.zeros((self.whole_size[1], self.whole_size[0],3), dtype=np.uint8)   # HW3
        arr[:self.dsize[1], :self.dsize[0], :] = frame1_s
        arr[:self.dsize[1], (self.dsize[0] + self.margin):, :] = frame2_s
        arr[(self.dsize[1] + self.margin):, :self.dsize[0], :] = frame3_s

        return arr

    def write_3_frames(self, frame_ls, stamp):
        """ 弃 """
        arr = self.combine_3_frames(frame_ls, stamp)
        self.videoWriter.write(arr)

    def write_4_frames(self, frame_ls, img_dic, stamp):
        arr = self.combine_3_frames(frame_ls, stamp)
        img_ls, id_ls, camid_ls = [], [], []
        for k, v_dic in img_dic.items():
            id_ls.append(k)
            camid_ls.append(v_dic['camid'])
            img_ls.append(cv2.resize(v_dic['img'], self.patch_size))

        if len(img_ls) <= self.grid_size[0]:
            row, col = 2, 7
        elif len(img_ls) <= self.grid_size[1]:
            row, col = 4, 14
        elif len(img_ls) <= self.grid_size[2]:
            row, col = 8, 28
        elif len(img_ls) <= self.grid_size[3]:
            row, col = 16, 56
        else:
            print('too many gallery imgs, only show the recent 50 imgs')
            ind = np.argsort(id_ls)[::-1][:50]   # 从大到小的索引, 取前50个
            img_ls = [x for i, x in enumerate(img_ls) if i in ind]
            id_ls = list(np.array(id_ls)[ind])
            camid_ls = list(np.array(camid_ls)[ind])
            row, col = 4, 14

        fig, axes = plt.subplots(row, col, figsize=(3 * col, 6 * row))
        plt.clf()        # 这里先清理画布的分区(画布的大小其实保留了), 稍后重新添加分区(单个添加)
        for i, img in enumerate(img_ls):
            ax = fig.add_subplot( row, col, i+1) 
            ax.imshow(img)
            ax.set_title('ID%d-Cam%d'%(id_ls[i], camid_ls[i]), fontsize=22)
            ax.axis("off")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        plt.close()
        frame4 = np.array(Image.open(buf))
        frame4_s = cv2.resize(frame4, self.dsize)
        arr[(self.dsize[1] + self.margin):, (self.dsize[0] + self.margin):, :] = frame4_s
        self.videoWriter.write(arr)

    def show_info(self):
        print('allert_num: box0-box1-reid = %d-%d-%d'%(self.allert_box_0, self.allert_box, self.allert_reid))


class Warp_predictor():
    def __init__(self, cfg, input_size=(128,384), device='0'):
        """ input_size: w*h """
        self.predictor = DefaultPredictor(cfg)
        self.color_candidate = [(255, 0, 255), (0, 255,0), (0, 255, 255)]
        self.input_size = input_size
        self.device = device

    @staticmethod
    def plot_one_box(x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    def get_patches_by_cam(self, boxes_ls, frame_ls):
        patches_ls = [[],[],[]]
        i = 0
        for boxes, frame in zip(boxes_ls, frame_ls):
            # boxes对应一个cam的frame中检测到的行人坐标集合
            if boxes.shape[0]:
                img_c = frame.copy()
                for *xyxy, conf, _ in boxes:
                    # save pedestrain patch
                    x1,y1,x2,y2 = int(xyxy[0].item()), int(xyxy[1].item()), int(xyxy[2].item()), int(xyxy[3].item())
                    patch_img = img_c[y1:y2, x1:x2, :]
                    patches_ls[i].append(patch_img)

                    # plot bbox on the whole img ==> 就地修改frame
                    w_ = abs(x2 - x1)
                    h_ = abs(y2 - y1)
                    label = '%.2f %d*%d' % (conf, h_, w_)
                    self.plot_one_box(xyxy, frame, label=label, color=self.color_candidate[0], line_thickness=3)
            i += 1

        return patches_ls

    def preprocess(self, patches_ls):
        blob = []
        for patches in patches_ls:
            for patch in patches:
                patch_ = cv2.resize(patch, self.input_size)
                blob.append(patch_[:,:,::-1])     # bgr2rgb
        blob = np.array(blob, dtype=np.float32)   # nhwc
        blob = torch.from_numpy(blob).permute(0,3,1,2)   # nchw; 无须显示转device
        
        return blob

    def postprocess(self, feats):
        pass

    def extract_patches(self, patches_ls):
        input_blob = self.preprocess(patches_ls)
        feats = self.predictor(input_blob).numpy()
        
        return feats


class Warp_detector():
    def __init__(self, device='0'):
        self.detector = YoloHand(size=1280, device=device)

    def detect_3_frames(self, frame_ls):
        boxes_ls = []
        for frame in frame_ls:
            boxes = self.detector.detect(frame)
            boxes_ls.append(boxes)

        return boxes_ls


class Gallery():
    def __init__(self, thresh_sim=0.90, thresh_stamp=2000):
        self.pool_dic = {}                  # 存放gallery特征 
        self.timestamp = {}                 # 存放gallery特征的时间戳
        self.img_dic = {}                   # 存放gallery特征最新的图片块及其cam_id
        self.id_num = 1                     # 添加gallery特征时, 用作键
        self.thresh_sim = thresh_sim        # 相似度阈值
        self.thresh_stamp = thresh_stamp    # 时间戳阈值

    def create(self, feats, stamp, patches_ls):
        """ 初始建库
            feats: re_id features, np.ndarray, num*dim;
            stamp: time stamp, int num;
            patches_ls: pedestrain image patches, list of list of np.ndarray;
        """
        # 以下考虑相邻相机有overlap时的初始建库，暂取消
        # sim = np.matmul(feats, feats.T)
        # sim = np.triu(sim, k=1)       # 取上三角阵，不含对角线
        # x_arr, y_arr = np.where(sim > self.thresh)

        # 将初始的所有特征用于建库，因开始的重复ID不影响报警次数
        patch_ls, camid_ls = self.get_patches(patches_ls)
        for i in range(feats.shape[0]):
            self.pool_dic[self.id_num] = feats[i].copy()
            self.timestamp[self.id_num] = stamp
            self.img_dic[self.id_num] = {'img':patch_ls[i], 'camid':camid_ls[i]}
            self.id_num += 1

    @staticmethod
    def get_patches(patches_ls):
        """ 将[[a],[b],[c]]转为[a,b,c] """
        patch_ls = []
        camid_ls = []   # camid: {1,2,3}
        for i, patches in enumerate(patches_ls):
            for patch in patches:
                patch_ls.append(patch)
                camid_ls.append(i+1)

        return patch_ls, camid_ls

    def gallery_feats(self):
        """ 将dict类型的gallery特征转换为list,方便转为np.ndarray """
        key_ls = []
        feat_ls = []
        for k, v in self.pool_dic.items():
            key_ls.append(k)
            feat_ls.append(v)

        return key_ls, feat_ls  # 

    def match_update(self, query_feats, stamp, patches_ls):
        """ 进行query和gallery的匹配, 及gallery的更新 """
        # match_ls = [0] * len(query_feats)
        key_ls, feat_ls = self.gallery_feats()
        sim = np.matmul(query_feats, np.array(feat_ls).T)
        q_ind = np.where( np.max(sim, axis=1) > self.thresh_sim )[0]    # np.ndarray
        g_ind = np.argmax(sim, axis=1)[q_ind]   
        
        patch_ls, camid_ls = self.get_patches(patches_ls)
        # 更新匹配上query的gallery特征
        for i, j in zip(g_ind, q_ind):
            key = key_ls[i]
            # match_ls[j] = 1
            self.pool_dic[key] = 0.5*( self.pool_dic[key] + query_feats[j] ) 
            self.timestamp[key] = stamp
            self.img_dic[key] = {'img':patch_ls[j], 'camid':camid_ls[j]}
            
        # 添加未匹配的query特征至gallery
        for i in range(len(query_feats)):
            if i not in q_ind:
                self.pool_dic[self.id_num] = query_feats[i]
                self.timestamp[self.id_num] = stamp
                self.img_dic[self.id_num] = {'img':patch_ls[i], 'camid':camid_ls[i]}
                self.id_num += 1

        # 整合输出结果
        match_dic = {'q_ind': q_ind, 
                     'g_ind': g_ind, 
                     'all_match': q_ind.shape[0]==query_feats.shape[0] 
                    }
        
        return match_dic   # match_ls

    def check_stamp(self, stamp, verbose=True):
        """ 删除过期的gallery ID """
        key_ls = []
        for k, v in self.timestamp.items():
            if stamp - v > self.thresh_stamp:
                key_ls.append(k)
        for k in key_ls:
            self.timestamp.pop(k)
            self.pool_dic.pop(k)
            self.img_dic.pop(k)
        if verbose and key_ls:
            print('delete %d ID: '%(len(key_ls)), key_ls)

        return key_ls

    def show_info(self, stamp):
        """ 显示gallery信息 """
        print('timestamp: [%d]  \nID num: [%d]' % (stamp, len(self.pool_dic)))
        info_str = ''
        count = 0
        for k, v in self.timestamp.items():
            count += 1
            if count % 3:
                info_str += '\tID_%d: timestamp_%d ;' % (k, v)
            else:
                info_str += '\tID_%d: timestamp_%d ;\n' % (k, v)
        if count % 3 == 0:
            info_str = info_str[:-1]  # 若能整除，去掉换行符
        print(info_str)
        print('='*50)

