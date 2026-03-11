import math
import time
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import matplotlib.pyplot as plt
import os
from models.projector import TemporalContrastiveLoss
import torch.nn.functional as F


def train_one_epoch(model: torch.nn.Module, aux_decoder: torch.nn.Module, projector: torch.nn.Module, mask_adapter: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, tubelet_size=1, wandb_logger=None, 
                    teacher_model=None, embedding_weight=0.8, past_future_weight=20):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = nn.MSELoss(reduction="none")
    #loss_func = nn.L1Loss(reduction="none")# 改：L1
    embedding_loss = nn.CosineEmbeddingLoss(margin=0.15, reduction='mean')
    contrastive_loss = TemporalContrastiveLoss(temperature=0.5)
    
    temperature = 0.5  # 可调 0.07~0.2

    def ce_align(student, teacher, T=0.07):
        """
        student, teacher: [B, N, C]  （如 fs/ft_vis 或 past_mask/ft_mask）
        返回：InfoNCE 风格的 CE 损失
        """
        s = F.normalize(student, dim=-1)   # 保持和“余弦”等价的尺度
        t = F.normalize(teacher, dim=-1)
        B, N, C = s.shape
        # [B, N, N] 每个样本内，两两相似度；对角线是正样本
        logits = torch.matmul(s, t.transpose(1, 2)) / T
        labels = torch.arange(N, device=student.device).unsqueeze(0).expand(B, N)  # [B, N]
        loss = F.cross_entropy(logits.reshape(-1, N), labels.reshape(-1))
        return loss

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        
        # split_past_current_future
        B, C, T, H, W = videos.shape
        T_each = T // 3
        videos_past   = videos[:, :, :T_each]             # [B, C, T/3, H, W]
        videos_current = videos[:, :, T_each:2*T_each]    # [B, C, T/3, H, W]
        videos_future = videos[:, :, 2*T_each:]           # [B, C, T/3, H, W]
                
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos_current * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=tubelet_size, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=tubelet_size, p1=patch_size, p2=patch_size)
            
            #print(f"mask.shape: {bool_masked_pos.shape}")
            B, _, C = videos_patch.shape
            # labels = videos_patch[bool_masked_pos].reshape(B, -1, C) # torch.Size([4, 2352, 768])
            
            # encode_past_future
            _, past_fs, _ = model(videos_past, None, tubelet_size)
            _, future_fs, _ = model(videos_future, None, tubelet_size)
            
            if teacher_model is not None:
                # ft = teacher_model(videos).detach()
                ft = teacher_model(videos_current).detach()
                #print(ft.shape)
                B, N, C = ft.shape
                # each_N = N//3
                # ft = ft[:,each_N:each_N*2,:]
                
                # ft_vis = ft[~bool_masked_pos].reshape(B, -1, C)
                # ft_mask = ft[bool_masked_pos].reshape(B, -1, C)

        loss = 0
        loss_embedding = 0
        with torch.cuda.amp.autocast():
            
            # mask guidence
            # p_x, _, bool_masked_pos = mask_adapter(videos_current,teacher_features = ft)
            p_x, _, bool_masked_pos = mask_adapter(videos_current)
            
            C = 384  # student channel dim
            # 把 teacher token 特征投到 student 维度
            # if teacher_model is not None:
            #     assert ft is not None
            #     ft = align_proj(ft)    # [B, N, Cs]
            
            labels = videos_patch[bool_masked_pos].reshape(B, -1, 768) # torch.Size([4, 2352, 768])
            ft_vis = ft[~bool_masked_pos].reshape(B, -1, C)
            ft_mask = ft[bool_masked_pos].reshape(B, -1, C)
            
            outputs, fs, all_fs = model(videos_current, bool_masked_pos, tubelet_size) # torch.Size([4, 2352, 1536])
            
            # auxiliary reconstruction
            past, past_cross_attn_weights, past_self_attn_weights = aux_decoder(past_fs, all_fs)
            future, future_cross_attn_weights, future_self_attn_weights = aux_decoder(past_fs, all_fs)
            past_mask = past[bool_masked_pos].reshape(B, -1, C)
            future_mask = future[bool_masked_pos].reshape(B, -1, C)
            loss_mse = torch.mean(loss_func(input=outputs, target=labels),dim=-1)# Reconstruction loss:B, N_m (mask tokens)
             
            # contrastive learning
            h_past, h_current, h_future = projector(past_fs, fs, future_fs, past_attn=past_cross_attn_weights, future_attn=future_cross_attn_weights)
            loss_contrast = contrastive_loss(h_past, h_current, h_future)
            loss += loss_contrast * 0.05
            
            # Sampling loss: l_s -> B, N_m
            l_s =torch.zeros(B, ).to(loss_mse.device)
            for i in range(p_x.shape[0]):
                # categorical distribution
                m = torch.distributions.categorical.Categorical(probs=p_x[i])
                # log-probabilities
                log_probs = m.log_prob(torch.arange(0, p_x.shape[1], 1).to(p_x.device)) # 1, N_m
                # mask log-probs
                mask_log_probs = log_probs[bool_masked_pos[i]]
                # we need to select tokens that maximize the reconstruction error, so (-) sign
                l_s[i] = -torch.mean(mask_log_probs*loss_mse[i].detach())
            
            loss += torch.mean(loss_mse) + 1e-4*torch.mean(l_s)#Reconstruction loss & Sampling loss
            target = torch.ones(past_mask.shape[0], past_mask.shape[1]).to(device)
            loss_pf = torch.mean(loss_func(input=past_mask, target=future_mask))
            loss += loss_pf * past_future_weight
            if teacher_model is not None:
                loss_embedding_past = embedding_loss(past_mask.reshape(-1, past_mask.shape[-1]),
                                                        ft_mask.reshape(-1, ft_mask.shape[-1]),
                                                        target.view(-1))
                loss_embedding_future = embedding_loss(future_mask.reshape(-1, future_mask.shape[-1]),
                                                        ft_mask.reshape(-1, ft_mask.shape[-1]),
                                                        target.view(-1))
                target = torch.ones(fs.shape[0], fs.shape[1]).to(device)
                loss_embedding_current = embedding_loss(fs.reshape(-1, fs.shape[-1]),
                                                        ft_vis.reshape(-1, ft_vis.shape[-1]),
                                                        target.view(-1))
                # # Cross-Entropy Loss
                # loss_embedding_past    = ce_align(past_mask,   ft_mask,   T=temperature)
                # loss_embedding_future  = ce_align(future_mask, ft_mask,   T=temperature)
                # loss_embedding_current = ce_align(fs,          ft_vis,    T=temperature)
                # loss_embedding = loss_embedding_past + loss_embedding_future
                # loss += loss_embedding * embedding_weight + loss_embedding_current

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger is not None:
            wandb_logger.set_step()
            wandb_logger.update(head="scalar", loss=loss, embedding=loss_embedding)
            # Log outputs and labels as images
            if step % 1000 == 0: 
                patch_idx = 0
                videos_s = IMAGENET_DEFAULT_STD
                videos_m = IMAGENET_DEFAULT_MEAN

                label_patch = labels[0, patch_idx, :].detach().cpu().numpy()  # [768]
                label_patch = label_patch.reshape(patch_size, patch_size, 3)  # [16, 16, 3]
                # label_patch = label_patch.transpose(1, 2, 0)  # [16, 16, 3]
                label_patch = np.clip(label_patch / videos_s + videos_m, 0, 1)
                label_patch = (label_patch * 255).astype(np.uint8)

                output_patch = outputs[0, patch_idx, :].detach().cpu().numpy()  # [768]
                output_patch = output_patch.reshape(patch_size, patch_size, 3)  # [16, 16, 3]
                # output_patch = output_patch.transpose(1, 2, 0)  # [16, 16, 3]
                output_patch = np.clip(output_patch / videos_s + videos_m, 0, 1)
                output_patch = (output_patch * 255).astype(np.uint8)

                wandb_logger.update_image(head="image", outputs=output_patch, labels=label_patch)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    