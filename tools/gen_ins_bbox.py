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

def inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, img_path_files, mode):
    for img_path in tqdm(img_path_files):
        src_img_path = os.path.join(kitti_data_dir, img_path)
        dst_bbox_path = img_path.replace("png", "txt").replace('kitti_data', 'kitti_data_bbox_eigen_zhou/'+mode)
        dst_img_path = img_path.replace("png", "npy").replace('kitti_data', 'kitti_data_ins_eigen_zhou/'+mode)
        if os.path.exists(dst_bbox_path) == True and os.path.exists(dst_img_path) == True:
            continue
        else:
            print(dst_bbox_path)
            print(dst_img_path)

        rgb_img = np.array(Image.open(src_img_path).convert('RGB'))
        # crop and convert RGB to BGR
        rgb_img = rgb_img[:, :, ::-1].copy() 
        output = predictor(rgb_img)
        # --------------------------------------------------------------
        bbox = output['instances'].to("cpu").pred_boxes.tensor.numpy()
        make_dir(os.path.dirname(dst_bbox_path))
        np.savetxt(dst_bbox_path, bbox, newline="\n")
        # --------------------------------------------------------------
        mask = output['instances'].pred_masks.cpu().numpy().transpose([1,2,0])
    
        ins_class = output['instances'].pred_classes.cpu().numpy()
        ins_1_0 = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.int8)
        ins_1_1 = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.int8)

        for i, sig_class in enumerate(ins_class):
            ins_1_0[mask[:,:,i]] = sig_class+1

        for i, sig_class in enumerate(ins_class):
            ins_1_1[mask[:,:,i]] = i+1
        
        ins_pack_0 = np.expand_dims(ins_1_0, axis=2)
        ins_pack_1 = np.expand_dims(ins_1_1, axis=2) 

        ins_cat = np.concatenate((ins_pack_0, ins_pack_1), axis=2)

        make_dir(os.path.dirname(dst_img_path))
        np.save(dst_img_path, ins_cat)
        # --------------------------------------------------------------

if __name__ == "__main__":
    # NOTE: change mode, achine_code and file_names here
    mode = "png"
    '''
    machine_code = "/userhome/34/h3567721/"
    train_files = os.path.join(machine_code, "monodepth-project/monodepth2/splits/eigen_zhou/train_files.txt")
    val_files = os.path.join(machine_code, "monodepth-project/monodepth2/splits/eigen_zhou/val_files.txt")
    test_files = os.path.join(machine_code, "monodepth-project/monodepth2/splits/eigen/test_files.txt")
    '''
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
    make_dir(kitti_data_ins_dir)
    make_dir(kitti_data_bbox_dir)

    train_lines = get_img_path(kitti_data_dir, train_lines, mode)
    val_lines = get_img_path(kitti_data_dir, val_lines, mode)
    test_lines = get_img_path(kitti_data_dir, test_lines, mode)   

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    print("inference test")
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, test_lines, mode="test")
    print("inference val")
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, val_lines, mode="val")
    print("inference train")
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, train_lines, mode="train")

