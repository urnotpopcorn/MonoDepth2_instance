# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks

def exe_time(func):
	def new_func(*args, **args2):
		t0 = time.time()
		# print("@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__))
		back = func(*args, **args2)
		# print("@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__))
		print("@%.3fs taken for {%s}" % (time.time() - t0, func.__name__))
		return back
	return new_func

class Trainer:
    def __init__(self, options):
        self.opt = options
        
        print("---------------------------------------------------------------")
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #if self.opt.project_dir == "/userhome/34/h3567721/monodepth-project":
        #    os.environ["CUDA_VISIBLE_DEVICES"] = "0" # user #0 GPU Card on CS GPU Farm
        
        print("Current project dir: ", self.opt.project_dir)
        #print("CUDA_VISIBlE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
        print("torch.cuda.is_available()", torch.cuda.is_available())  
        print("---------------------------------------------------------------")

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        # log_dir: /userhome/34/h3567721/monodepth-project/tmp
        # model_name: M_640x192_instance_pose
        # log_path: /userhome/34/h3567721/monodepth-project/tmp/M_640x192_instance_pose

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        self.num_scales = len(self.opt.scales) # default: 4
        self.num_input_frames = len(self.opt.frame_ids) # default: 3 [0, -1, 1]
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames     # 3

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"


        # ------------------------------------------------------------------------------------------
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        
        # TODO:
        if self.opt.depth_train:
            self.parameters_to_train += list(self.models["encoder"].parameters())
        else:
            for p in self.models["encoder"].parameters():
                p.requires_grad = False

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        # TODO:
        if self.opt.depth_train:
            self.parameters_to_train += list(self.models["depth"].parameters())
        else:
            for p in self.models["depth"].parameters():
                p.requires_grad = False

        # ------------------------------------------------------------------------------------------

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        
        self.models["pose_encoder"].to(self.device)
        
        # FIXME: the flag
        if not self.opt.fix_pose:
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        else:
            if self.opt.pose_train:
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            else:
                for p in self.models["pose_encoder"].parameters():
                    p.requires_grad = False
            '''
            pose_encoder_para = self.models["pose_encoder"].parameters()
            for p in pose_encoder_para:
                p.requires_grad = False
            
            self.parameters_to_train += list(pose_encoder_para)
            '''

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)

        self.models["pose"].to(self.device)

        if not self.opt.fix_pose:
            self.parameters_to_train += list(self.models["pose"].parameters())
        else:
            if self.opt.pose_train:
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            else:
                for p in self.models["pose_encoder"].parameters():
                    p.requires_grad = False

            '''
            pose_decoder_para = self.models["pose"].parameters()
            for p in pose_decoder_para:
                p.requires_grad = False
            
            self.parameters_to_train += list(pose_decoder_para)
            '''
        
        # --------------------------------------------------------------------------------
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            # 也可以判断是否为conv2d，使用相应的初始化方式 
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # 是否为批归一化层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if self.opt.instance_pose:
            self.models["instance_pose"] = networks.InsPoseDecoder(
                num_RoI_cat_features=1024,
                num_input_features=1,
                num_frames_to_predict_for=2)
            
            self.models["instance_pose"].apply(weight_init)
            self.models["instance_pose"].to(self.device)
            '''
            instance_pose_para = self.models["instance_pose"].parameters()
            self.parameters_to_train += list(instance_pose_para)
            '''
            # TODO:
            if self.opt.ins_pose_train:
                self.parameters_to_train += list(self.models["instance_pose"].parameters())

        # --------------------------------------------------------------------------------
        # 1e-4 -> 1e-5 (15 epoch) -> 1e-6 (30 epoch), step=15
        #self.model_optimizer = optim.Adam(
        #    filter(lambda p: p.requires_grad, self.parameters_to_train), self.opt.learning_rate)
        
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # loal local model to initialize the network
        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        # default: kitti
        self.dataset = datasets_dict[self.opt.dataset]

        # default: eigen_zhou
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train")) # 39810
        val_filenames = readlines(fpath.format("val"))     # 4424
        # test_filenames = readlines(fpath.format("test"))   # 4424

        # --png for our settings.
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)

        # 39810/12*20=66350 iterations, 3317 iterations per epoch
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        # --------------------------------------------------------------------------------
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 
            num_scales=4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 
            num_scales=4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # test_dataset = self.dataset(
        #     self.opt.data_path, test_filenames, self.opt.height, self.opt.width, self.opt.frame_ids, 
        #     num_scales=4, is_train=False, img_ext=img_ext, opt=self.opt, mode="test")
        # self.test_loader = DataLoader(
        #     test_dataset, self.opt.batch_size, True,
        #     num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        # self.test_iter = iter(self.test_loader)

        # --------------------------------------------------------------------------------
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # add no_ssim flag => use SSIM
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales: # [0,1,2,3]
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # initialize backproject_depth and project_3d at each scale
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items \n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        # --------------------------------------------------------------------------------

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        # default 20 epochs
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            # save model every 1 epoch
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you
        # should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  
        # Failure to do this will result in PyTorch skipping the first value of the learning rate schedule.
        # See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if torch.__version__ >= "1.1.0":
            pass
        else:
            self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            # print("-------------------------------")
            # print("batch_idx: ", batch_idx)
            before_op_time = time.time()
            
            outputs, losses = self.process_batch(inputs)
            
            self.model_optimizer.zero_grad()

            # start_time = time.time()
            losses["loss"].backward()
            # print("@" + str(round(time.time()-start_time, 4)) + " taken for {loss-backward}")

            self.model_optimizer.step()
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            #early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            #late_phase = self.step % 2000 == 0
            
            early_phase = batch_idx % self.opt.log_frequency == 0
            #if early_phase or late_phase:
            if early_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses["ins_loss"].cpu().data, losses["bg_loss"].cpu().data)
                
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                
                self.log("train", inputs, outputs, losses)
                #self.val()
            self.step += 1
        
        # self.test()

        if torch.__version__ >= "1.1.0":
            self.model_lr_scheduler.step()
        else:
            pass        

    def test(self):
        """test the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.test_iter.next()
        except StopIteration:
            self.test_iter = iter(self.test_loader)
            inputs = self.test_iter.next()
            

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("test", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()
            
        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    # @exe_time
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device) 
            
        
        if self.opt.warp_direction:
            disp_net_input = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 0)
            # input: [bs*3, 1, 192, 640], frame 0 -1 1 are cat in dim 0
            # output: [[bs*3, 64, 96, 320], [bs*3, 64, 48, 160], [bs*3, 128, 24, 80], [bs*3, 256, 12, 40], [bs*3, 512, 6, 20]]
            features = self.models["encoder"](disp_net_input)
        else:
            # input: [bs, 1, 192, 640] # only frame 0
            # output: [[bs, 64, 96, 320], [bs, 64, 48, 160], [bs, 128, 24, 80], [bs, 256, 12, 40], [bs, 512, 6, 20]]
            features = self.models["encoder"](inputs["color_aug", 0, 0])
        
        outputs = self.models["depth"](features)

        outputs.update(self.predict_poses(inputs))

        if self.opt.instance_pose:
            self.generate_images_pred(inputs, outputs)
            self.synthesize_layer(inputs, outputs)
            # FIXME: need to improve
            losses = self.compute_losses(inputs, outputs)
            weight_fg, weight_bg, ins_losses = self.compute_instance_losses(inputs, outputs)
            if ins_losses['ins_loss'].detach().cpu().numpy() == np.nan:
                print('nan')
                input()

            losses['ins_loss'] = ins_losses['ins_loss']

            bg_loss = losses['loss']
            fg_loss = losses['ins_loss']
            losses['bg_loss'] = bg_loss
            
            #weight_bg = 0.5
            #weight_fg = 0.5
            if self.opt.weight_fg is not None:
                losses['loss'] = (1-self.opt.weight_fg) * bg_loss + self.opt.weight_fg * fg_loss
            else:
                losses['loss'] = weight_bg * bg_loss + weight_fg * fg_loss
            #losses['loss'] += ins_losses['ins_loss']

            return outputs, losses
        else:
            self.generate_images_pred(inputs, outputs) # warp t-1 to t and t+1 to t
            losses = self.compute_losses(inputs, outputs)
            return outputs, losses

    # @exe_time
    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.
            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]: # [-1, 1]
                # To maintain ordering we always pass frames in temporal order
                if self.opt.warp_direction:
                    if f_i < 0:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]] # 0, -1
                    else:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]] # 1, 0
                else:
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]] # -1, 0
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]] # 0, 1

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                # pose decoder
                axisangle, translation = self.models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                
                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    # @exe_time
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """

        # ([('disp', 3), ('disp', 2), ('disp', 1), ('disp', 0)])
        # ([bs, 1, 24, 80], [bs, 1, 48, 160], [bs, 1, 96, 320],[bs, 1, 192, 640]) or 
        # ([bs*3, 1, 24, 80],[bs*3, 1, 48, 160], [bs*3, 1, 96, 320],[bs*3, 1, 192, 640])
        for scale in self.opt.scales: # [0,1,2,3]
            disp = outputs[("disp", scale)]

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            # min_depth = 0.1, max_depth = 100
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            if self.opt.warp_direction:
                bs = self.opt.batch_size
                for i, frame_id in enumerate(self.opt.frame_ids): # [0, -1, 1]
                    # [bs*3, 1, 192, 640] -> [bs, 1, 192, 640] for each frame_id 
                    outputs[("depth", frame_id, scale)] = depth[i*bs:(i+1)*bs,:, :, :]

                for i, frame_id in enumerate(self.opt.frame_ids[1:]): # [-1, 1]
                    T = outputs[("cam_T_cam", 0, frame_id)] # T from -1 to 0 or from 1 to 0
                    # cam_points of frame -1 or 1, am_points = [12, 4, 122880]
                    cam_points = self.backproject_depth[source_scale](
                        outputs[("depth", frame_id, scale)], inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)

                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", 0, source_scale)],
                        pix_coords,
                        padding_mode="border")

                    # if use automasking
                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]
            else:
                outputs[("depth", 0, scale)] = depth
                for i, frame_id in enumerate(self.opt.frame_ids[1:]): # [-1, 1]
                    T = outputs[("cam_T_cam", 0, frame_id)] # T from 0 to -1 or 0 to 1
                    # cam_points of frame 0, [12, 4, 122880]
                    cam_points = self.backproject_depth[source_scale](
                        depth, inputs[("inv_K", source_scale)])
                    pix_coords = self.project_3d[source_scale](
                        cam_points, inputs[("K", source_scale)], T)
                    outputs[("sample", frame_id, scale)] = pix_coords

                    # default samping method: the out-of-boundary pixel will be assigned as border value
                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, source_scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")

                    # if use automasking
                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, source_scale)]

    # @exe_time
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        bs = self.opt.batch_size

        for scale in self.opt.scales: # [0, 1, 2, 3]
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            if self.opt.warp_direction:
                # TODO:
                pass
            else:
                disp = outputs[("disp", scale)]
                color = inputs[("color", 0, scale)] # tgt frame 
                target = inputs[("color", 0, source_scale)] # tgt frame

                for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                    # src_{t-1, t+1} is warped to tgt frame
                    # compare the loss between tgt and warped_src
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                # if use automasking
                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                # if use automasking
                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).to(self.device) * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                # if use automasking
                if not self.opt.disable_automasking:
                    outputs["identity_selection/{}".format(scale)] = \
                        (idxs > identity_reprojection_loss.shape[1] - 1).float()

                loss += to_optimise.mean()

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss += loss
                losses["loss/{}".format(scale)] = loss
        
        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses
    
    def extract_bboxes(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (x1, y1, x2, y2)].
        """

        # [b, h, w]
        mask = ins_warp_mask.squeeze(1)

        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            idx_mask = mask[bs_idx, :, :].detach().cpu().numpy() # [h, w]
            # Bounding box.
            horizontal_indicies = np.where(np.any(idx_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance
                x1, y1, x2, y2 = 0, 0, 20, 6

            ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))

        # [[1, 4]*bs]
        return ins_warp_bbox

    def extract_bboxes_mine(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        # ins_warp_mask: [bs, 1, 192, 640]
        mask = ins_warp_mask.squeeze(1)
        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            #idx_mask = mask[bs_idx, :, :].uint8()#.detach().cpu().numpy()
            # Bounding box.
            idx_mask = mask[bs_idx, :, :].type(torch.uint8)
            horizontal_indicies = torch.where(torch.any(idx_mask, axis=0))[0]
            vertical_indicies = torch.where(torch.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, y1, x2, y2 = 0, 0, 20, 6
            ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))
            #ins_warp_bbox.append([[x1, y1, x2, y2]])
            #ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))
        
        # list of [1,4]
        return ins_warp_bbox
        
    # @exe_time
    def synthesize_layer(self, inputs, outputs):
        if self.opt.warp_direction:
            # TODO:
            pass
        else:
            # warp -1 to 0 and warp 1 to 0
            scale = 0 # only care about the scale 0
            bs = self.opt.batch_size
            
            # compute cam_points of current frame
            disp = outputs[("disp", scale)] # 0
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # min_depth = 0.1, max_depth = 100
            cam_points = self.backproject_depth[scale](depth, inputs[("inv_K", scale)]) # t
            
            # compute mask of current frame
            ins_one_hot_0 = inputs[("ins_id_seg", 0, scale)]    # [bs, K+1, 192, 640] of frame 0
            tgt_dynamic_layer = torch.sum(ins_one_hot_0[:, 1:, :, :], 1).unsqueeze(1).float()  # [bs, 1, 192, 640], exclude bg
            total_mask = tgt_dynamic_layer
            
            # compute numbers of instance
            instance_K_num = ins_one_hot_0.shape[1] - 1         # K+1-1 = K

            # compute dynamic area of current frame
            outputs[("f_img_syn", 0, scale)] = inputs["color", 0, scale] * tgt_dynamic_layer
            
            for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                # compute camera pose of static area
                T_static = outputs[("cam_T_cam", 0, frame_id)] # [bs, 4, 4]
                
                # get image of frame_id 
                total_img_frame_id = inputs[("color", frame_id, scale)]
                
                # get mask of frame_id, in tgt_dynamic_layer: 0 stands for bg, 1 stands for object instance
                total_mask_frame_id = torch.sum(inputs[("ins_id_seg", frame_id, scale)][:, 1:, :, :], 1).unsqueeze(1).float()  # [bs, 1, 192, 640], exclude bg
                
                # define the final image and mask and RoI
                f_img_syn = torch.zeros_like(total_img_frame_id) # final image
                f_mask_syn = torch.zeros_like(total_mask) # final mask
                #f_RoI_syn = torch.zeros_like(total_mask)

                # extract features, feed warped_t-1_2_t or warped_t+1_2_t into depth encoder
                ins_RoI_bbox_frame_id = inputs[("ins_RoI_bbox", frame_id, scale)] # [bs, k ,4]
                ins_RoI_bbox_list_frame_id = [x.squeeze(0) for x in list(ins_RoI_bbox_frame_id.split(1, dim=0))] # [bs, k ,4] -> [[K, 4]*bs]
                f_feats_0_frame_id = self.models["encoder"](inputs["color_aug", frame_id, scale])[-1] # [bs, 512, 6, 20]
                # [k*bs, 512, 6, 20] -> [k*bs, 512, 3, 3]
                cur_RoI_feats = torchvision.ops.roi_align(f_feats_0_frame_id, ins_RoI_bbox_list_frame_id, output_size=(3,3))

                for ins_id in range(instance_K_num):
                    # step1: get image feature of frame 0 and frame frame_id
                    f_feats_0 = self.models["encoder"](inputs["color_aug", 0, scale])[-1] # [bs, 512, 6, 20]
                    f_feats_0_frame_id = self.models["encoder"](inputs["color_aug", frame_id, scale])[-1] # [bs, 512, 6, 20]

                    # step2: predict mask of frame 0 from frame frame_id 
                    pix_coords = self.project_3d[scale](cam_points, inputs[("K", scale)], T_static) #[b, h, w, 2]
                    ins_mask_frame_id = inputs[("ins_id_seg", frame_id, scale)][:, ins_id+1, :, :].unsqueeze(1).float() #[b, 1, h, w]
                    ins_warp_mask = F.grid_sample(ins_mask_frame_id, pix_coords) #[b, 1, h, w]

                    # step3: predict bbox of frame 0 from frame frame_id 
                    ins_warp_bbox = self.extract_bboxes(ins_warp_mask)
                    
                    # TODO: step3.5: compute diff between t_pred and t_gt and then eliminate relative static area
                    roi_diff = torch.sum(torch.abs(outputs[("color", frame_id, scale)] * ins_warp_mask - inputs["color", 0, scale] * ins_mask_frame_id))
                    if torch.sum(ins_warp_mask) >= 1:
                        roi_diff = roi_diff / (torch.sum(ins_warp_mask))
                        if self.opt.roi_diff_thres is not None and roi_diff < self.opt.roi_diff_thres:
                            continue

                    # step4: use mask to crop ROI feature in frame frame_id 
                    ins_cur_RoI_feats = torch.cat([cur_RoI_feats[i*instance_K_num+ins_id, :, :, :].unsqueeze(0) for i in range(bs)])

                    # step5: use warped bbox to crop ROI feature in frame 0
                    ins_0_RoI_feats = torchvision.ops.roi_align(f_feats_0, ins_warp_bbox, output_size=(3,3)) # [b, 512, 3, 3]

                    # step6: input ins_pose_net and predict T_dynamic
                    if frame_id < 0:
                    #if False:
                        ins_pose_inputs = [ins_cur_RoI_feats, ins_0_RoI_feats] # -1, 0
                    else:
                        ins_pose_inputs = [ins_0_RoI_feats, ins_cur_RoI_feats] # 0, 1
                    
                    ins_pose_inputs = torch.cat(ins_pose_inputs, 1)
                    axisangle, translation = self.models["instance_pose"](ins_pose_inputs)
                    cam_t_cam = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(frame_id < 0)) # [bs, 4, 4]
                    #cam_t_cam = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False) # [bs, 4, 4]
                    ins_cam_T_cam = list(torch.chunk(cam_t_cam, self.opt.batch_size, dim=0))   # [[K, 4, 4] * bs]

                    #T_dynamic = torch.cat([x.unsqueeze(0) for x in ins_cam_T_cam], 0) # [bs, K, 4, 4]
                    T_dynamic = torch.cat([x for x in ins_cam_T_cam], 0) # [bs, K, 4, 4]
                    
                    # step7: for each instance, generate a warped image and a warped mask
                    T_total = torch.matmul(T_dynamic, T_static) # [bs, 4, 4]
                    pix_coords = self.project_3d[scale](cam_points, inputs[("K", scale)], T_total)
                    
                    # step8: predict frame 0 from frame frame_id based on T_dynamic and T_static
                    ins_warp_img = F.grid_sample(total_img_frame_id, pix_coords)
                    ins_warp_mask = F.grid_sample(ins_mask_frame_id, pix_coords)

                    # step9: add image and mask 
                    # support render_by_depth later
                    f_img_syn = torch.add(f_img_syn*(1-ins_warp_mask), ins_warp_img*ins_warp_mask)
                    f_mask_syn = torch.add(f_mask_syn*(1-ins_warp_mask), ins_warp_mask)

                # store for visualization
                # ins-wise warped image from frame frame_id. 0 stands for bg
                outputs[("f_img_syn", frame_id, scale)] = f_img_syn
                
                # update output image
                color_ori = outputs[("color", frame_id, scale)]
                color_new = f_mask_syn * f_img_syn + (1-f_mask_syn) * color_ori

                outputs[("color_ori", frame_id, scale)] = color_ori
                outputs[("color_diff", frame_id, scale)] = color_new - color_ori
                outputs[("color", frame_id, scale)] = color_new
                outputs[("warped_mask", frame_id, scale)] = f_mask_syn
                #outputs[("ori_mask", frame_id, scale)] = total_mask_frame_id
            
            outputs[("cur_mask", 0, scale)] = tgt_dynamic_layer
                   

    # @exe_time
    def compute_instance_losses(self, inputs, outputs):
        """loss of dynamic region"""
        bs = self.opt.batch_size

        if self.opt.warp_direction:
            #TODO:
            pass
        else:
            losses = {}
            scale = 0

            # get mask of current frame
            total_mask = outputs[("cur_mask", 0, scale)]
            
            if total_mask.sum() < 1:
                # all static
                # TODO: need to improve
                losses["ins_loss"] = torch.zeros(1, requires_grad=True).mean().to(self.device)
                weight_fg = 0
                weight_bg = 1 - weight_fg
            else:
                color = inputs[("color", 0, scale)]
                disp = outputs[("disp", scale)]
                disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)

                smooth_loss = get_smooth_loss(norm_disp, color)
                smooth_loss =  self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

                # print("    smooth_loss: ", smooth_loss)
                losses["ins_loss/{}_smooth_loss".format(scale)] = smooth_loss

                reprojection_losses = []
                tgt_dynamic = outputs[("f_img_syn", 0, 0)]
                for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                    pred_dynamic = outputs[("f_img_syn", frame_id, 0)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred_dynamic, tgt_dynamic))

                combined = torch.cat(reprojection_losses, 1)
                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                # TODO: need to improve
                #reproj_loss = to_optimise.mean()
                reproj_loss = to_optimise.sum() / total_mask.sum()
                
                # print("    reproj_loss: ", reproj_loss)
                losses["ins_loss/{}_reproj".format(scale)] = reproj_loss
                losses["ins_loss_{}".format(scale)] = reproj_loss + smooth_loss

                # TODO: scale
                losses["ins_loss"] = reproj_loss + smooth_loss
                #print("repro\tsmooth\t", reproj_loss.detach().cpu().numpy(), smooth_loss.detach().cpu().numpy())

                weight_fg = total_mask.sum() / total_mask.nelement()
                weight_bg = 1 - weight_fg

        return weight_fg, weight_bg, losses
    
    # @exe_time
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)  # [12,3,192,640]
        l1_loss = abs_diff.mean(1, True)     # [12,1,192,640]
        
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        # cap 80
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss, ins_loss=None, bg_loss=None):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        
        if ins_loss is not None:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f}| ins_loss: {:.5f} | bg_loss: {:.5f} | time elapsed: {} | time left: {}"
            
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, ins_loss, bg_loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"
            
            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, add_image=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        # save each loss term
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        if add_image ==True:
            for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
                s = 0
                # for s in self.opt.scales:
                for frame_id in self.opt.frame_ids: # 0, -1, 1
                    writer.add_image(
                        "color_input_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    
                    
                    if frame_id != 0:
                        # warped image
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)
                    
                    if self.opt.instance_pose:
                        if self.opt.warp_direction:  
                            # if lucky, inputs_f_img_syn_-1 should be close to outputs_f_img_syn_-1
                            # or: inputs_f_img_syn_1 should be close to outputs_f_img_syn_1
                            if frame_id != 0:
                                # dynamic layer of original t-1 or t+1
                                writer.add_image(
                                    "inputs_f_img_syn_{}_{}/{}".format(frame_id, s, j),
                                    inputs[("f_img_syn", frame_id, 0)][j].data, self.step)
                                # dynamic layer from warp_t_2_t-1 and warp_t_2_t+1 
                                writer.add_image(
                                    "outputs_f_img_syn_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("f_img_syn", frame_id, 0)][j].data, self.step)
                        else:
                            # if lucky, outputs_f_img_syn_-1 shoubld be close to outputs_f_img_syn_0
                            # or outputs_f_img_syn_1 shoubld be close to outputs_f_img_syn_0
                            # 0: dynamic layer from t
                            # -1: dynamic layer from warp_t-1_2_t
                            # 1: dynamic layer from warp_t+1_2_t
                            if frame_id != 0:
                                writer.add_image(
                                    "color_pred_ori_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color_ori", frame_id, s)][j].data, self.step)
                                
                                writer.add_image(
                                    "outputs_f_img_syn_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("f_img_syn", frame_id, 0)][j].data, self.step)
                                
                                writer.add_image(
                                    "color_diff_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("color_diff", frame_id, 0)][j].data, self.step)

                                writer.add_image(
                                    "warped_mask_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("warped_mask", frame_id, 0)][j].data, self.step)
                                
                                #writer.add_image(
                                #    "ori_mask_{}_{}/{}".format(frame_id, s, j),
                                #    outputs[("ori_mask", frame_id, 0)][j].data, self.step)
                                
                                '''
                                outputs[("color_ori", frame_id, scale)] = color_ori
                                outputs[("color_diff", frame_id, scale)] = color_new - color_ori
                                outputs[("color", frame_id, scale)] = color_new
                                outputs[("mask", frame_id, scale)] = f_mask_syn
                                '''
                        '''
                        for ins_id in range(4):
                            writer.add_image(
                                "RoI_img_0_{}/{}/{}".format(s, j, ins_id),
                                outputs[("RoI_img_0", frame_id, s, ins_id)][j].data, self.step)
                        '''

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                
                writer.add_image(
                    "cur_mask_{}/{}".format(s, j),
                    outputs[("cur_mask", 0, s)][j].data, self.step)
                
                

                    
                # if using automasking
                # if not self.opt.disable_automasking:
                #     writer.add_image(
                #         "automask_{}/{}".format(s, j),
                #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)
            
        writer.close()

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        # ~/model_name/models/opt.json
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        # "~/tmp/model_name/weights_#epoch/"
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        # adam.pth, depth.pth, encoder.pth, pose.pth, pose_encoder.pth

    def load_model(self):
        """Load model(s) from disk
        """
        # ~/tmp/M_640x192/models/weights_9
        # /userhome/34/h3567721/tmp/M_640x192/models/weights_9
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        # default models_to_load: ["encoder", "depth", "pose_encoder", "pose"]
        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        if self.opt.fix_pose or self.opt.fix_depth:
            pass
        else:
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            if os.path.isfile(optimizer_load_path):
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
            else:
                print("Cannot find Adam weights so Adam is randomly initialized")
