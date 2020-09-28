import os
import cv2
import numpy as np
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

from tqdm import tqdm

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_img_path(kitti_data_dir, img_files, mode):
    if mode == "png":
        img_ext = ".png"
    elif mode == "jpg":
        img_ext = ".jpg"

    img_path_files = []
    for img_path in img_files:
        # 2011_09_30/2011_09_30_drive_0028_sync 2300 r
        frame_instant = int(img_path.split(" ")[1])
        t_frame_0 = str(frame_instant - 1)
        t_frame_1 = str(frame_instant)
        t_frame_2 = str(frame_instant + 1)
        img_index_0 = t_frame_0.zfill(10)
        img_index_1 = t_frame_1.zfill(10)
        img_index_2 = t_frame_2.zfill(10)

        if img_path.split(" ")[2] == "l":
            img_dir = "02"
        elif img_path.split(" ")[2] == "r":
            img_dir = "03"

        t_frame_0_src = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_0)+img_ext)
        t_frame_1_src = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_1)+img_ext)
        t_frame_2_src = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_2)+img_ext)

        if os.path.exists(t_frame_0_src):
            img_path_files.append(t_frame_0_src)
        
        if os.path.exists(t_frame_1_src):
            img_path_files.append(t_frame_1_src)

        if os.path.exists(t_frame_2_src):
            img_path_files.append(t_frame_2_src)

    return list(set(img_path_files))

def check(kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, img_path_files, mode):
    num = 0
    for img_path in tqdm(img_path_files):
        src_img_path = os.path.join(kitti_data_dir, img_path)
        if os.path.exists(src_img_path) == False:
            print(src_img_path)
            input()

        dst_bbox_path = img_path.replace("png", "txt").replace('kitti_data', 'kitti_data_bbox_eigen_zhou/'+mode)
        dst_img_path = img_path.replace("png", "npy").replace('kitti_data', 'kitti_data_ins_eigen_zhou/'+mode)
        if os.path.exists(dst_bbox_path) == False:
            num += 1
    return num

if __name__ == "__main__":
    # NOTE: change mode, achine_code and file_names here
    mode = "png"
    train_files = "/mnt/sdb/xzwu/Code/MonoDepth2_inverse/splits/eigen_zhou/train_files.txt"
    val_files = "/mnt/sdb/xzwu/Code/MonoDepth2_inverse/splits/eigen_zhou/val_files.txt"
    test_files = "/mnt/sdb/xzwu/Code/MonoDepth2_inverse/splits/eigen/test_files.txt"

    with open(train_files, "r") as fd:
        train_lines = fd.read().splitlines()

    with open(val_files, "r") as fd:
        val_lines = fd.read().splitlines()

    with open(test_files, "r") as fd:
        test_lines = fd.read().splitlines()

    # /userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/2011_09_26_drive_0001_sync_02/***.jpg
    #base_dir = os.path.join(machine_code, "monodepth-project")
    base_dir = "/mnt/sdb/xzwu/Code/MonoDepth2_inverse"
    kitti_data_dir = os.path.join(base_dir, "kitti_data")
    kitti_data_ins_dir = os.path.join(base_dir, "kitti_data_ins_eigen_zhou")
    kitti_data_bbox_dir = os.path.join(base_dir,  "kitti_data_bbox_eigen_zhou")

    train_lines = get_img_path(kitti_data_dir, train_lines, mode)
    val_lines = get_img_path(kitti_data_dir, val_lines, mode)
    test_lines = get_img_path(kitti_data_dir, test_lines, mode)   

    print("check test")
    num = check(kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, test_lines, mode="test")
    print(num)
    input()
    print("check val")
    num = check(kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, val_lines, mode="val")
    print(num)
    input()
    print("check train")
    num = check(kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, train_lines, mode="train")
    print(num)

