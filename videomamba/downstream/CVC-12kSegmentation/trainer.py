import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from dataset import *
import medpy

import os, json
import torch
import numpy as np
from PIL import Image

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def fp_fn_tp_to_color(pred_mask: torch.Tensor, gt_hw: torch.Tensor) -> np.ndarray:
    """
    二类分割的 FN / FP / TP 可视化：
      FN (gt=1, pred=0) -> yellow
      FP (gt=0, pred=1) -> red
      TP (gt=1, pred=1) -> green
    其它 -> black
    """
    # 如果是多类，把 >0 当作前景
    gt = (gt_hw > 0).detach().cpu().numpy().astype(np.uint8)
    pred = (pred_mask > 0).detach().cpu().numpy().astype(np.uint8)

    h, w = gt.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    fn = (gt == 1) & (pred == 0)   # 假阴
    fp = (gt == 0) & (pred == 1)   # 假阳
    tp = (gt == 1) & (pred == 1)   # 真阳

    vis[fn] = [255, 255, 0]  # yellow
    vis[fp] = [255,   0, 0]  # red
    vis[tp] = [  0, 255, 0]  # green

    return vis


def _denorm_to_uint8(img_chw: torch.Tensor) -> np.ndarray:
    """img_chw: (C,H,W) normalized. C==3 -> ImageNet反归一化；否则做min-max并凑成3通道。"""
    x = img_chw.detach().float()
    C = x.shape[0]
    if C == 3:
        x = (x * _IMAGENET_STD.to(x.device) + _IMAGENET_MEAN.to(x.device)).clamp(0, 1)
        return (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    # 非3通道：min-max到[0,1]
    x = x - x.min()
    denom = x.max()
    x = x / denom if float(denom) > 0 else torch.zeros_like(x)
    if C == 1:
        x3 = x.repeat(3,1,1)
    elif C >= 3:
        x3 = x[:3]
    else:
        x3 = x.repeat(3,1,1)[:3]
    return (x3.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

def _mask_to_uint8(mask_hw: torch.Tensor) -> np.ndarray:
    """(H,W) int/long -> 灰度uint8。二类0/255，多类线性映射到0..255。"""
    m = mask_hw.detach().cpu().numpy().astype(np.uint8)
    vmax = int(m.max())
    if vmax <= 1:
        return (m * 255).astype(np.uint8)
    scale = 255 // vmax if vmax > 0 else 1
    return (m * scale).astype(np.uint8)

def _to_5d_video(img_4d: torch.Tensor) -> torch.Tensor:
    """
    统一成 (1, C, T, H, W) —— 模型的输入布局（通道在前）。
    支持原始 (C,T,H,W) 或 (T,C,H,W) 两种。
    """
    assert img_4d.ndim == 4, f"unexpected shape {img_4d.shape}"
    # (C,T,H,W)
    if img_4d.shape[0] in (1, 3):
        return img_4d.unsqueeze(0)  # (1,C,T,H,W)
    # (T,C,H,W)
    if img_4d.shape[1] in (1, 3):
        return img_4d.permute(1, 0, 2, 3).unsqueeze(0)  # -> (1,C,T,H,W)
    # 兜底：按 (C,T,H,W)
    return img_4d.unsqueeze(0)


def _select_frame_index(T: int, policy):
    """policy: 'center' | 'first' | 'last' | int"""
    if isinstance(policy, int):
        return max(0, min(T-1, policy))
    if policy == 'first':  return 0
    if policy == 'last':   return T-1
    return min(T//2, T-1)  # default center

@torch.no_grad()
def save_fixed_test_visuals(model, dataset, device, snapshot_path, epoch_idx,
                            vis_indices, frame_policy='center'):
    """
    从测试集 dataset 中按固定 vis_indices 抽样；对每个样本取指定帧，保存原图/GT/Pred。
    输出目录: {snapshot_path}/visuals/epoch_{epoch:03d}/
    文件名: s{rank}_idx{ds_idx}_t{t}_img/gt/pred.png
    """
    out_dir = os.path.join(snapshot_path, "visuals", f"epoch_{epoch_idx:03d}")
    os.makedirs(out_dir, exist_ok=True)

    # 记录选择的索引，确保可追溯
    sel_path = os.path.join(snapshot_path, "visuals", "selection.json")
    if not os.path.exists(sel_path):
        os.makedirs(os.path.dirname(sel_path), exist_ok=True)
        with open(sel_path, "w") as f:
            json.dump({"vis_indices": vis_indices, "note": "fixed test indices for visualization"}, f, indent=2)

    was_training = model.training
    model.eval()

    for rank, ds_idx in enumerate(vis_indices):
        sample = dataset[ds_idx]  # 需要 dataset.__getitem__ 返回 {'image','label'}
        img4 = sample['image'].to(device)   # (C,T,H,W) 或 (T,C,H,W)
        lab  = sample['label'].to(device)   # (T,H,W) 或可能带通道的一维

        # 统一形状
        if lab.ndim == 4 and lab.shape[0] == 1:
            lab = lab.squeeze(0)            # (T,H,W)
        assert img4.ndim == 4, f"image shape {img4.shape}"
        assert lab.ndim == 3, f"label shape {lab.shape}"

        img5 = _to_5d_video(img4)          # (1,C,T,H,W)
        B, C, T, H, W = img5.shape
        t_idx = _select_frame_index(T, frame_policy)

        # 前向：现在送入的是 (1,C,T,H,W)
        logits_btchw = model(img5)         # 期望返回 (T, num_classes, H, W) 或 (B*T, ...)
        # 为稳妥，兼容两种返回
        if logits_btchw.ndim == 4 and logits_btchw.shape[0] in (T, B*T):
            # 如果是 (B*T, C', H, W)，先还原到 (T, C', H, W)（因为 B=1）
            if logits_btchw.shape[0] == B * T:
                logits_tchw = logits_btchw.view(B, T, *logits_btchw.shape[1:]).squeeze(0)
            else:
                logits_tchw = logits_btchw  # (T, C', H, W)
        else:
            raise ValueError(f"Unexpected model output shape: {logits_btchw.shape}")

        # 取第 t_idx 帧
        img_chw   = img5[0, :, t_idx]                 # (C,H,W)  —— 注意这里通道在前
        gt_hw     = lab[t_idx]                        # (H,W)
        pred_log  = logits_tchw[t_idx]                # (C',H,W)
        pred_mask = torch.argmax(torch.softmax(pred_log, dim=0), dim=0)
        
        # === 新增：生成 FN/FP/TP 颜色可视化 ===
        conf_vis = fp_fn_tp_to_color(pred_mask, gt_hw)

        # 保存
        stem = f"s{rank:02d}_idx{ds_idx:04d}_t{t_idx:03d}"
        Image.fromarray(_denorm_to_uint8(img_chw)).save(os.path.join(out_dir, f"{stem}_img.png"))
        Image.fromarray(_mask_to_uint8(gt_hw)).save(os.path.join(out_dir, f"{stem}_gt.png"))
        Image.fromarray(_mask_to_uint8(pred_mask)).save(os.path.join(out_dir, f"{stem}_pred.png"))
        Image.fromarray(conf_vis).save(os.path.join(out_dir, f"{stem}_fpfn_tp.png"))

    # 还原训练/评估状态
    if was_training:
        model.train()


def save_epoch_visuals(image_batch, label_bt, outputs_bt, epoch_idx, out_root):
    """
    兼容两种视频布局:
      - (B, T, C, H, W)
      - (B, C, T, H, W)
    并与训练中的展平顺序 out_idx=b*T+t 一致。
    """
    os.makedirs(os.path.join(out_root, "visuals", f"epoch_{epoch_idx:03d}"), exist_ok=True)
    out_dir = os.path.join(out_root, "visuals", f"epoch_{epoch_idx:03d}")

    # 一点点健壮性检查
    assert outputs_bt.ndim == 4, f"Unexpected outputs shape: {outputs_bt.shape}"
    assert label_bt.ndim == 3, f"Unexpected label shape: {label_bt.shape}"

    # 取 b=0、t=中间帧
    b_idx = 0

    if image_batch.ndim == 5:
        # 可能是 (B,T,C,H,W) 或 (B,C,T,H,W)
        # 通过“哪个维度在 {1,3} 中”来判断通道轴
        if image_batch.shape[2] in (1, 3):
            # (B, T, C, H, W)
            B, T, C, H, W = image_batch.shape
            t_idx = min(T // 2, T - 1)
            out_idx = b_idx * T + t_idx
            img_chw = image_batch[b_idx, t_idx]         # (C,H,W)
        elif image_batch.shape[1] in (1, 3):
            # (B, C, T, H, W)
            B, C, T, H, W = image_batch.shape
            t_idx = min(T // 2, T - 1)
            out_idx = b_idx * T + t_idx
            img_chw = image_batch[b_idx, :, t_idx]      # (C,H,W)
        else:
            # 兜底：按 (B,T,C,H,W) 处理
            B, T, C, H, W = image_batch.shape
            t_idx = min(T // 2, T - 1)
            out_idx = b_idx * T + t_idx
            img_chw = image_batch[b_idx, t_idx]
    elif image_batch.ndim == 4:
        # (B,C,H,W) —— 没有时间维，按单帧处理
        B, C, H, W = image_batch.shape
        T = 1
        t_idx = 0
        out_idx = 0
        img_chw = image_batch[0]
    else:
        raise ValueError(f"Unexpected image_batch shape: {image_batch.shape}")

    # 与 label/outputs 的展平对齐
    assert out_idx < label_bt.shape[0], f"Index {out_idx} out of range for labels {label_bt.shape}"
    assert label_bt.shape[0] == outputs_bt.shape[0], \
        f"labels {label_bt.shape[0]} != outputs {outputs_bt.shape[0]} on dim 0"

    # 原图
    img_uint8 = _denorm_to_uint8(img_chw)
    Image.fromarray(img_uint8).save(os.path.join(out_dir, "img.png"))

    # GT（label_batch 已经 view 成 (B*T,H,W) 了）
    gt_hw = label_bt[out_idx]
    gt_uint8 = _mask_to_uint8(gt_hw)
    Image.fromarray(gt_uint8).save(os.path.join(out_dir, "gt.png"))

    # Pred（outputs_bt: (B*T, num_classes, H, W)）
    pred_logits = outputs_bt[out_idx]
    pred_mask = torch.argmax(torch.softmax(pred_logits, dim=0), dim=0)
    pred_uint8 = _mask_to_uint8(pred_mask)
    Image.fromarray(pred_uint8).save(os.path.join(out_dir, "pred.png"))
    # === 新增：FN/FP/TP 颜色可视化 ===
    conf_vis = fp_fn_tp_to_color(pred_mask, gt_hw)
    Image.fromarray(conf_vis).save(os.path.join(out_dir, "fpfn_tp.png"))




def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = medpy.metric.binary.dc(pred, gt)
        hd95 = medpy.metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1., 0.
    else:
        return 0., 0.


def eval_dice(pred_y, gt_y, classes=5):
    pred_y = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)
    all_dice = []
    all_hd95 = []

    pred_y = pred_y.cpu().detach().numpy()
    gt_y = gt_y.cpu().detach().numpy()

    for cls in range(1, classes):
        dice, hd95 = calculate_metric_percase(pred_y == cls, gt_y == cls)

        all_dice.append(dice)
        all_hd95.append(hd95)

    all_dice = torch.tensor(all_dice).cuda()
    all_hd95 = torch.tensor(all_hd95).cuda()

    return torch.mean(all_dice), torch.mean(all_hd95)


def eval(model, val_loader, device, classes):
    all_dice = []
    all_hd95 = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, label = batch['image'].to(device).squeeze(), batch['label'].to(device).squeeze()
            img = img.unsqueeze(0).permute(0, 2, 1, 3, 4)
            output = model(img)

            dice, hd95 = eval_dice(output, label, classes=classes)
            all_dice.append(dice.item())
            all_hd95.append(hd95.item())

    return np.mean(np.array(all_dice)), np.mean(np.array(all_hd95))


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, dynamic_padding_collate_fn
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu

    train_transform = transforms.Compose([
        Resize((args.img_size + 32, args.img_size + 32)),
        RandomCrop((args.img_size, args.img_size)),
        RandomFlip(),
        RandomRotation(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    db_train = SegCTDataset(dataroot=args.root_path, mode='train', transforms=train_transform)
    db_test = SegCTDataset(dataroot=args.root_path, mode='test', transforms=test_transform)
    
    # 可视化配置（没有就给默认值）
    vis_every  = getattr(args, "vis_every", 5)   # 每隔多少个epoch保存一次
    vis_count  = getattr(args, "vis_count", 20)    # 每次保存多少个测试样本
    vis_frame  = getattr(args, "vis_frame", "center")  # 选帧策略: 'center'/'first'/'last'/int

    # 固定测试样本索引（默认为前 vis_count 个）
    vis_indices = list(range(min(vis_count, len(db_test))))


    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn,collate_fn=dynamic_padding_collate_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    if args.test:
        model.eval()
        model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cpu'))
        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
        print('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))
        exit(0)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)

    if args.wandb:
        wandb.init(project='synapse-segmentation', config=args, dir="/gaoxieping/dsh/wandb/")
        wandb.watch(model, log='all')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_dice = 0.
    best_hd95 = 0.
    iterator = range(max_epoch)
    for epoch_num in iterator:
        model.train()

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            image_batch = image_batch.permute(0, 2, 1, 3, 4)
            outputs = model(image_batch)
            
            # print("Shape:", label_batch.shape)
            # print(outputs.shape)
            
            label_batch = label_batch.view(-1, outputs.shape[2], outputs.shape[3])

            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            if args.wandb:
                wandb.log({'lr': lr_, 'total_loss': loss, 'loss_ce': loss_ce})

            if iter_num % 20 == 0 and args.wandb:
                index = 0
                image = image_batch[0, :, index].permute(1, 2, 0).cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                wandb.log({'train/Image': [wandb.Image(image)]})
                outputs = torch.argmax(torch.softmax(outputs[index], dim=0), dim=0, keepdim=False).unsqueeze(0)
                outputs = outputs.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                wandb.log({'train/Prediction': [wandb.Image(outputs)]})
                labs = label_batch[index]
                labs = labs.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy()
                wandb.log({'train/GroundTruth': [wandb.Image(labs)]})

        test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)


        if test_dice > best_dice:
            best_dice = test_dice
            best_hd95 = test_hd95
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
            logging.info("save best model to best_model.pth")

        print('Epoch [%3d/%3d], Loss: %.4f, Dice: %.1f, HD95: %.1f, Best Dice: %.1f, Best HD95: %.1f' %
              (epoch_num + 1, max_epoch, loss.item(), test_dice * 100., test_hd95, best_dice * 100., best_hd95))
        
        # === 从测试集固定样本保存可视化 ===
        # if (epoch_num + 1) % vis_every == 0:
        #     try:
        #         save_fixed_test_visuals(
        #             model=model,
        #             dataset=db_test,
        #             device='cuda',
        #             snapshot_path=snapshot_path,
        #             epoch_idx=epoch_num + 1,
        #             vis_indices=vis_indices,
        #             frame_policy=vis_frame
        #         )
        #         logging.info(f"[VIS][test] saved {len(vis_indices)} samples at epoch {epoch_num + 1}")
        #     except Exception as e:
        #         logging.warning(f"[VIS][test] failed at epoch {epoch_num + 1}: {e}")



        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            torch.save(model.state_dict(), os.path.join(snapshot_path, 'last_model.pth'))
            logging.info("save last model to last_model.pth")
            break
    
    if args.wandb:
        wandb.finish()
    return "Training Finished!"
