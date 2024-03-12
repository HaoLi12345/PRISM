import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.optim import AdamW, lr_scheduler
from src.config.config_setup import build_model, get_dataloader
from monai.losses import DiceCELoss, DiceLoss
import torch.nn as nn
from src.utils.util import save_checkpoint, _bbox_mask
from src.utils import scribble, boundary_selection
import time
import os
import torch.distributed as dist
from torch.cuda import amp
import torchio as tio


class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        a = time.time()
        use_small = True if self.args.use_small_dataset else False
        self.train_data, self.val_data = get_dataloader(args, split='train', use_small=use_small), get_dataloader(args, split='val', use_small=use_small)
        if self.args.use_sam3d_turbo:
            self.sam = build_model(args, checkpoint='/home/hao/Hao/3D_medical_foundation_model/src/ckpt/sam_med3d_turbo.pth')
        else:
            self.sam = build_model(args)
        if self.args.ddp:
            self.sam = self.sam.module

        self.best_dice, self.best_epoch, self.start_epoch = 0, 0, 0
        self.pooling_layer = nn.AvgPool3d((self.args.boundary_kernel_size, self.args.boundary_kernel_size, 1), stride=1,
                                     padding=(int((self.args.boundary_kernel_size - 1) / 2),
                                              int((self.args.boundary_kernel_size - 1) / 2),
                                              0)).cuda()

        self.setup()
        print('dataloaders are created, models are loaded, and others are set, spent {} for rank {}'
              .format(round(time.time() - a, 2), self.args.rank))


    def run(self):
        self.scaler = amp.GradScaler()
        for epoch_num in range(self.start_epoch, self.args.max_epoch):
            self.sam.train()
            if self.args.ddp:
                # dist.barrier() # set a barrier until all processes are at same point
                self.train_data.sampler.set_epoch(epoch_num)

            self.train(epoch_num)
            if self.args.ddp and self.args.rank == 0:
                print('doing validation on rank=0')
                current_mean_dice = self.validate(epoch_num) if self.args.data != 'lits' else self.validater_sliding_window(epoch_num)
            else:
                current_mean_dice = self.validate(epoch_num) if self.args.data != 'lits' else self.validater_sliding_window(epoch_num)
            # https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
            # if self.args.ddp:
                # dist.barrier()
            self.save_model(current_mean_dice, epoch_num)

    def validate(self, epoch_num):
        self.sam.eval()
        return self.validater(epoch_num)



    def validater_sliding_window(self, epoch_num):
        with torch.no_grad():
            dice_summary = []
            for idx, (subject_dict, image_path) in enumerate(self.val_data):
                if subject_dict['label']['data'][0].sum() <= 0:
                    self.logger.info(image_path, 'label volume too small, and it has been skipped for validation')
                    continue
                mean_dice = 0
                subject = tio.Subject(image=tio.ScalarImage(tensor=subject_dict['image']['data'][0].float(), affine=subject_dict['image']['affine'][0]),
                                      label=tio.LabelMap(tensor=subject_dict['label']['data'][0].float(), affine=subject_dict['label']['affine'][0]))
                grid_sampler = tio.inference.GridSampler(subject, 128, 16)
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
                aggregator = tio.inference.GridAggregator(grid_sampler)

                masks_final = torch.zeros([self.args.iter_nums, len(patch_loader), 128, 128, 128])
                location_list = []
                for idx_patch, patches_batch in enumerate(patch_loader):
                    image, label = patches_batch['image'][tio.DATA].to(self.args.device), patches_batch['label'][tio.DATA].to(self.args.device)
                    locations = patches_batch[tio.LOCATION]

                    if torch.count_nonzero(label) == 0:
                        print('found empty patch')
                        masks = torch.zeros([self.args.iter_nums, 1, 128, 128, 128])
                    else:
                        _, masks = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=False, return_each_iter=True)
                        print(masks.shape)
                    masks_final[:, idx_patch, :] = masks.squeeze(1)
                    location_list.append(locations)

                mean_dice_sub_list = []
                for iter_num in range(self.args.iter_nums):
                    for l_i in range(0, len(location_list)):
                        location = location_list[l_i]
                        a = masks_final[iter_num, l_i, :].unsqueeze(0).unsqueeze(0)
                        mask = a
                        aggregator.add_batch(mask, location)
                    masks_iter_final = aggregator.get_output_tensor()
                    mean_dice_sub_list.append(self._get_dice_score(torch.sigmoid(masks_iter_final), subject.label.data))

                mean_dice_sub = np.mean(mean_dice_sub_list)
                mean_dice += mean_dice_sub
                dice_summary.append(mean_dice)
                print(mean_dice_sub)
                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_summary)))
        return dice_summary


    def validater(self, epoch_num):
        device = self.args.device
        with torch.no_grad():
            dice_summary = []
            for idx, (image, label, image_path, _) in enumerate(self.val_data):
                mean_dice = 0
                image, label = image.to(device), label.to(device)
                #with amp.autocast():

                if self.args.data == 'kits' and image.size(1) > 1:
                    label_final, masks_final = torch.zeros([1, 1, int(image.size(2) * 2), image.size(3), image.size(4)]), torch.zeros([self.args.iter_nums, 1, int(image.size(2) * 2), image.size(3), image.size(4)])

                    for channel_num in range(image.size(1)):
                        _, masks = self._interaction(self.sam, image[:, channel_num, :].unsqueeze(1), label[:, channel_num, :].unsqueeze(1), iter_nums=self.args.iter_nums, train=False, return_each_iter=True)
                        start_point, end_pont = 0 + channel_num * image.size(2), image.size(2) + channel_num * image.size(2)

                        masks_final[:, 0, start_point: end_pont, :] = masks[:, 0, :]
                        label_final[0, 0, start_point: end_pont, :] = label[0, channel_num, :]

                    mean_dice_sub_list = []
                    for iter_num in range(self.args.iter_nums):
                        mean_dice_sub_list.append(self._get_dice_score(torch.sigmoid(masks_final[iter_num]), label_final[0]))
                    mean_dice_sub = np.mean(mean_dice_sub_list)
                else:
                    mean_dice_sub, masks = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=False)

                mean_dice += mean_dice_sub
                dice_summary.append(mean_dice)
                print(mean_dice_sub)
                self.logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.val_data)) +
                    ' subject: ' + str(image_path) + ' mean dice over clicks:' + str(mean_dice) +
                    ' stich left and right side (total size): ' + str(label.size(1)))
            self.logger.info("- Val metrics mean dice: " + str(np.mean(dice_summary)))
        return dice_summary


    def train(self, epoch_num):
        loss_summary = []
        for idx, (image, label, image_path) in enumerate(self.train_data):
            self.optimizer.zero_grad()

            # increase speed based on gradient accumulation
            # my_context = self.sam.no_sync if self.args.rank != -1 and idx % self.args.accumulation_steps != 0 else nullcontext
            # with my_context():
            image, label = image.to(self.args.device), label.to(self.args.device)
            with amp.autocast():
                loss, _ = self._interaction(self.sam, image, label, iter_nums=self.args.iter_nums, train=True)

            loss_summary.append(loss.detach().cpu().numpy())

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.sam.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print('epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                  + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                  + ": rank:" + str(self.args.rank))
            self.logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, self.args.max_epoch, idx, len(self.train_data))
                + ": loss:" + str(round(loss_summary[-1].flatten()[0], 4))
                + ": rank:" + str(self.args.rank))
        print('current lr: {}'.format(self.optimizer.param_groups[0]["lr"]))
        # If the first iteration creates NaN gradients (e.g. due to a high scaling factor and thus gradient overflow),
        # the optimizer.step() will be skipped and you might get this warning.
        self._update_lr(epoch_num, warm_up=self.args.warm_up)
        self.logger.info("- Train metrics: " + str(np.mean(loss_summary)))

    def get_next_click3D_torch_2(self, prev_seg, label, mode='train'): # prev_seg --> probability
        batch_points = []
        batch_labels = []

        pred_masks = (prev_seg > 0.5)
        true_masks = (label > 0)
        fn_masks = torch.logical_and(true_masks, torch.logical_not(pred_masks))
        fp_masks = torch.logical_and(torch.logical_not(true_masks), pred_masks)

        to_point_mask = torch.logical_or(fn_masks, fp_masks)


        # do_scribble = random.random()
        # sample_method = random.choice(['line', 'center', 'default'])
        sample_method = 'center'
        scribble_types = {
            'line': 'LineScribble',
            'center': 'CenterlineScribble',
            'default': 'ContourScribble'
        }

        def create_scribble_mask(scribble_type, data):
            scribble_object = getattr(scribble, scribble_type)()
            scribble_mask = scribble_object.batch_scribble(data).permute(1, 2, 3, 0)
            return scribble_mask > 0


        points_list = [len(torch.argwhere(to_point_mask[i])) for i in range(to_point_mask.size(0))]
        points_min = min(points_list)
        num_clicks = self.args.num_clicks if mode == 'train' else self.args.num_clicks_validation
        click_size = points_min if num_clicks > points_min else num_clicks
        dynamic_size = random.randint(1, click_size) if self.args.dynamic and mode == 'train' else click_size
        print(f"num_clicks {num_clicks} points_length: {points_min} dynamic_size: {dynamic_size}")

        for i in range(label.shape[0]):
            bp_list, bl_list = [], []
            points = torch.argwhere(to_point_mask[i])

            point_index = np.random.choice(len(points), size=dynamic_size, replace=False)
            points_select = points[point_index] # each row tensor([0, x, y, z]), size --> num_clicks x 4

            for click_index in range(dynamic_size):
                point = points_select[click_index]
                if fn_masks[i, 0, point[1], point[2], point[3]]:
                    is_positive = True
                else:
                    is_positive = False

                bp = point[1:].clone().detach().reshape(1, 1, 3)
                bl = torch.tensor([int(is_positive), ]).reshape(1, 1)
                bp_list.append(bp)
                bl_list.append(bl)

            if self.args.use_scribble:
                fg, bg_orig = fn_masks[i].permute(3, 0, 1, 2).float(), fp_masks[i].permute(3, 0, 1, 2).float()

                bbx = _bbox_mask(label[i, 0, :].unsqueeze(0))
                diff_ = 15
                i_min, i_max = bbx[:, :, 0], bbx[:, :, 3]
                j_min, j_max = bbx[:, :, 1], bbx[:, :, 4]
                k_min, k_max = bbx[:, :, 2], bbx[:, :, 5]
                if max(0, i_min - diff_) < min(i_max + diff_, 126):
                    i_min, i_max = max(0, i_min - diff_), min(i_max + diff_, 126)
                if max(0, j_min - diff_) < min(j_max + diff_, 126):
                    j_min, j_max = max(0, j_min - diff_), min(j_max + diff_, 126)
                if max(0, k_min - diff_) < min(k_max + diff_, 126):
                    k_min, k_max = max(0, k_min - diff_), min(k_max + diff_, 126)

                bg_mask = torch.zeros_like(bg_orig).permute(1, 2, 3, 0)
                bg_mask[:, i_min:i_max, j_min:j_max, k_min:k_max] = 1
                bg = bg_orig * bg_mask.permute(3, 0, 1, 2)
                print('filter out voxels: {}'.format(torch.count_nonzero(bg_orig) - torch.count_nonzero(bg)))


                scribble_type = scribble_types.get(sample_method, scribble_types['default'])
                scribble_mask_fg = create_scribble_mask(scribble_type, fg)

                limit_num = 500
                if torch.count_nonzero(scribble_mask_fg) >= limit_num + 50:
                    a = torch.argwhere(scribble_mask_fg).size(0) - limit_num
                    random_number = random.randint(0, a)
                    fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)[:, random_number: random_number + limit_num, :] # for computation only
                else:
                    fg_coors = torch.argwhere(scribble_mask_fg)[:, 1:].unsqueeze(0)

                fg_coors_label = torch.ones(1, fg_coors.size(1))
                bp_list.append(fg_coors)
                bl_list.append(fg_coors_label)


                scribble_mask_bg = create_scribble_mask(scribble_type, bg)
                if torch.count_nonzero(scribble_mask_bg) >= limit_num + 50: # dynamic_size is 50
                    a = torch.argwhere(scribble_mask_bg).size(0) - limit_num
                    random_number = random.randint(0, a)
                    bg_coors = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)[:, random_number: random_number + limit_num, :] # Fixme run this only or smallest_n below
                else:
                    bg_coors = torch.argwhere(scribble_mask_bg)[:, 1:].unsqueeze(0)

                bg_coors_label = torch.zeros(1, bg_coors.size(1))
                bp_list.append(bg_coors)
                bl_list.append(bg_coors_label)

            batch_points.append(torch.cat(bp_list, dim=1))
            batch_labels.append(torch.cat(bl_list, dim=1))

        # for scribble
        if self.args.use_scribble:
            smallest_n = min(tensor.size(1) for tensor in batch_labels)
            batch_points = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in batch_points]
            batch_labels = [tensor[:, :smallest_n] if tensor.size(1) > smallest_n else tensor for tensor in batch_labels]

        # # Check the shapes of the adjusted tensors
        # for i, tensor in enumerate(batch_points):
        #     print(f"Tensor {i + 1} shape: {tensor.shape}")

        print('First batch:   fn: {:.4f}, fp: {:.4f}, label 0: {}, label 1: {}'.format(
            torch.count_nonzero(fn_masks[0]) / torch.count_nonzero(true_masks[0]),
            torch.count_nonzero(fp_masks[0]) / torch.count_nonzero(true_masks[0]),
            str(batch_labels[0].numel() - torch.count_nonzero(batch_labels[0])),
            str(torch.count_nonzero(batch_labels[0]))
        )
        )
        print('--- ===================================== ---')
        print('--- above before model, below after model ---')
        print('--- ===================================== ---')
        return batch_points, batch_labels

    def get_points(self, prev_masks, gt3D, gt3D_box, mode='train'):
        mode = 'train' if mode else 'validation'

        batch_points, batch_labels = self.get_next_click3D_torch_2(prev_masks, gt3D, mode=mode)

        points_co = torch.cat(batch_points, dim=0).to(self.args.device) # b x num_clicks x 3
        points_la = torch.cat(batch_labels, dim=0).to(self.args.device) # b x num_clicks x 1

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_input = points_co
        labels_input = points_la

        bbox_coords = _bbox_mask(gt3D_box[:, 0, :], mode=mode, dynamic=self.args.dynamic_box).to(self.args.device) if self.args.use_box else None

        return points_input, labels_input, bbox_coords

    def batch_forward(self, sam_model, features, image_embedding, image, prev_masks, points=None, boxes=None):
        prev_masks = F.interpolate(prev_masks, scale_factor=0.25)
        features = [features[i].to(self.args.device) for i in range(0, len(features))]

        new_point_embedding, new_image_embedding = sam_model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=prev_masks,
            image_embeddings=image_embedding.to(self.args.device)
        )

        mask, iou_pred = sam_model.mask_decoder(
            prompt_embeddings=new_point_embedding,  # (B, 2, 256)
            image_embeddings=new_image_embedding,  # (B, 256, 64, 64)
            feature_list=features,
            image_level_features=None,
            image=image, points=[self.click_points, self.click_labels] if self.args.refine else [] # FIXME refine or not
        )
        return mask, iou_pred

    def _interaction(self, sam_model, image, label, iter_nums, train=False, return_each_iter=False):
        if return_each_iter:
            return_mask_total_iter = torch.zeros([iter_nums, 1, image.size(2), image.size(3), image.size(4)])

        image_embedding, feature_list = self.sam.image_encoder(image)
        self.click_points = []
        self.click_labels = []
        return_loss = 0
        prev_masks = torch.zeros_like(label, dtype=torch.float).to(label.device)
        for iter_num in range(iter_nums):

            loss = 0
            scale_factor = (iter_num + 1) if self.args.scale_loss else 1
            prev_masks_sigmoid = torch.sigmoid(prev_masks) if iter_num > 0 else prev_masks

            if self.args.init_learning and iter_num == 0:
                boundary, margin, content = boundary_selection.find_boundary_map(label)
                use_content = True
                for batch_index in range(label.size(0)):
                    if torch.count_nonzero(content[batch_index]) < self.args.num_clicks:
                        use_content = False
                if use_content:
                    label_sample = content
                else:
                    label_sample = label
            else:
                label_sample = label

            points_input, labels_input, box_input = self.get_points(prev_masks_sigmoid, label_sample, label, mode=train)
            mask, iou_pred = self.batch_forward(sam_model, feature_list, image_embedding, image, prev_masks, points=[points_input, labels_input], boxes=box_input)

            # ========================================================
            if self.args.multiple_outputs:
                iou_pred_best, max_label_index = torch.max(iou_pred, dim=1)
                mask_list = [mask[i, max_label_index[i], :].unsqueeze(0) for i in range(mask.size(0))]
                mask_best = torch.stack(mask_list, dim=0)
            else:
                mask_best = mask
                iou_pred_best = iou_pred

            # ========================================================
            if train:
                if self.args.multiple_outputs: # TODO: only back propagate lowest dice ? remove this after try
                    for i in range(mask.size(1)):
                        single_mask, single_dice = mask[:, i, :].unsqueeze(1), iou_pred[:, i]
                        loss += scale_factor * self._calculate_loss(single_mask, prev_masks, single_dice, label, labels_input, iter_num)
                else:
                    loss = self._calculate_loss(mask, prev_masks, iou_pred[:, 0], label, labels_input, iter_num)

                # ========================================================
                if self.args.refine:
                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                    print('dice before refine {} and after {}'.format(
                        self._get_dice_score(torch.sigmoid(mask_best), label),
                        self._get_dice_score(torch.sigmoid(mask_refine), label)))

                    # ========================================================
                    if self.args.refine_inter:
                        loss += scale_factor * self._calculate_loss(mask_refine, mask_best, iou_pred_best, label, labels_input, iter_num, inter=True)
                    else:
                        loss += scale_factor * self.loss_segmentation(mask_refine, label) * 1

                    mask_best = mask_refine  # FIXME refine or not, fixed return mask


                if self.args.init_learning and iter_num == 0:
                    loss = 10 * loss

            # ========================================================
            else:
                if self.args.refine:
                    mask_refine, error_map = self.sam.mask_decoder.refine(image, mask_best, [self.click_points, self.click_labels], mask_best.detach())
                    self.logger.info('dice before refine {} and after {}, label 0: {}, label 1: {}'.format(
                        self._get_dice_score(torch.sigmoid(mask_best), label), self._get_dice_score(torch.sigmoid(mask_refine), label),
                        str(labels_input.numel() - torch.count_nonzero(labels_input)), str(torch.count_nonzero(labels_input)) ) )
                    mask_best = mask_refine  # FIXME refine or not

                loss = self._get_dice_score(torch.sigmoid(mask_best), label) # dice

            return_loss += loss
            prev_masks = mask_best

            if return_each_iter:
                return_mask_total_iter[iter_num, :] = mask_best
        if return_each_iter:
            print(return_mask_total_iter.shape)
            return return_loss / iter_nums, return_mask_total_iter
        else:
            return return_loss / iter_nums, prev_masks

    def _calculate_loss(self, mask, prev_masks, pred_dice, label, labels_input, iter_num, inter=False):
        mask_probs, prev_masks_prob = torch.sigmoid(mask), torch.sigmoid(prev_masks)

        seg_edge = abs(label - self.pooling_layer(label))
        mask_edge = abs(mask_probs - self.pooling_layer(mask_probs))
        edge_number = torch.count_nonzero(mask_edge) + 1


        # or somewhere is named as iou, but it's dice
        pred_dice_score_loss = 0
        for batch_index in range(mask.size(0)):
            target_dice = 1 - self.loss_validation(mask[batch_index].unsqueeze(0), label[batch_index].unsqueeze(0))[0,0,0,0,0]

            target_dice = torch.tensor([target_dice])[0].to(self.args.device)
            pred_dice_score_loss += self.loss_boundary(pred_dice[batch_index], target_dice) * 1

        loss = self.loss_segmentation(mask, label) + self.loss_boundary(mask_edge, seg_edge) * 10
        loss = loss + pred_dice_score_loss
        return loss

    def save_model(self, current_dice, epoch_num):
        is_best = False
        if np.mean(current_dice) > self.best_dice:
            self.best_dice = np.mean(current_dice)
            self.best_epoch = epoch_num
            is_best = True

        if not self.args.ddp or (self.args.ddp and self.args.rank == 0):
            save_checkpoint({"epoch": epoch_num + 1,
                             "best_val_loss": self.best_dice,
                             "model_state_dict": self.sam.state_dict(),
                             "optimizer": self.optimizer.state_dict(),
                             "lr_scheduler": self.lr_scheduler.state_dict(),
                             },
                            is_best=is_best,
                            checkpoint=self.args.save_dir)
        self.logger.info("- Val metrics best mean dice: {} at epoch {} " .format(self.best_dice, self.best_epoch))



    def setup(self):
        self.setup_loss()
        self.setup_optimizier()
        self.setup_scheduler()

        if self.args.resume:
            if self.args.ddp:
                dist.barrier()
            checkpoint = 'best.pth.tar' if self.args.resume_best else 'last.pth.tar'
            ckpt = torch.load(os.path.join(self.args.save_dir, checkpoint))

            self.start_epoch = ckpt["epoch"]
            self.best_epoch = self.start_epoch
            self.best_dice = ckpt["best_val_loss"]
            self.sam.load_state_dict(ckpt["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            #self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])lr_scheduler_regular
            self.lr_scheduler_regular.load_state_dict(ckpt['lr_scheduler'])


            self.logger.info(f"Resume training from epoch {self.start_epoch}!")
            del ckpt
            torch.cuda.empty_cache()




    def setup_loss(self):
        self.loss_boundary = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction='none')

        self.loss_segmentation = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        self.loss_Dice = DiceLoss(sigmoid=True)
        self.loss_validation = DiceLoss(sigmoid=True, reduction='none')

        self.l1 = nn.L1Loss()
        self.inter_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


    def setup_optimizier(self):
        self.optimizer = AdamW([
            {'params': self.sam.image_encoder.parameters()},
            {'params': self.sam.prompt_encoder.parameters()},
            {'params': self.sam.mask_decoder.parameters()},
        ], lr=self.args.lr)

    def setup_scheduler(self):
        if self.args.lr_scheduler == 'linear':
            self.lr_scheduler_regular = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=500)
        else:
            self.lr_scheduler_regular = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
        if self.args.warm_up:
            self.linear_warmup_scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)

    def _update_lr(self, epoch, warmup_epoch=10, warm_up=False):
        if warm_up:
            if epoch < warmup_epoch:
                self.lr_scheduler = self.linear_warmup_scheduler
            else:
                self.lr_scheduler = self.lr_scheduler_regular
        else:
            self.lr_scheduler = self.lr_scheduler_regular
        self.lr_scheduler.step()

    def _get_dice_score(self, prev_masks, gt3D, batch=False):
        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        if batch:
            return dice_list
        else:
            return (sum(dice_list) / len(dice_list)).item()









