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
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

# =========================
#  Motion perturbations (Segmentation) - aligned with latest classification code
# =========================

RATIOS = [0.2, 0.5, 0.8]  # 20% / 50% / 80%
SEVERITY_TABLE = {
    0.2: {"tmax": 4,  "rmax": 5.0,  "smax": 3},   # p=20%
    0.5: {"tmax": 8, "rmax": 10.0, "smax": 6},   # p=50%
    0.8: {"tmax": 16, "rmax": 15.0, "smax": 10},  # p=80%
}

def _parse_ratio_from_name(name: str) -> float:
    # e.g. "temporal_p20" -> 0.2
    try:
        p_str = name.rsplit("_p", 1)[1]
        return float(p_str) / 100.0
    except Exception as e:
        raise ValueError(f"Bad perturb_name format: {name}. Expect like 'jitter_p20'") from e

def _get_severity_cfg(ratio: float):
    r = round(float(ratio), 1)
    if r not in SEVERITY_TABLE:
        raise ValueError(f"Unsupported ratio={ratio}. Supported: {list(SEVERITY_TABLE.keys())}")
    return SEVERITY_TABLE[r]

def _choose_indices(T, rng, ratio):
    k = int(round(T * float(ratio)))
    k = max(0, min(T, k))
    if k == 0:
        return set()
    return set(rng.choice(T, size=k, replace=False).tolist())

def _to_DCHW(img):
    """
    img: [D,C,H,W] or [C,D,H,W] (or rare [C,H,W])
    return: [D,C,H,W], layout flag
    """
    if img.dim() == 3:
        # treat as [C,H,W]
        return img, "CHW"
    if img.dim() != 4:
        raise ValueError(f"Expect 3D/4D image, got {tuple(img.shape)}")

    # likely [D,C,H,W]
    if img.shape[1] in (1, 3):
        return img, "DCHW"
    # likely [C,D,H,W]
    if img.shape[0] in (1, 3):
        return img.permute(1, 0, 2, 3).contiguous(), "CDHW"
    # fallback
    return img, "DCHW"

def _from_DCHW(img_dchw, layout):
    if layout == "CDHW":
        return img_dchw.permute(1, 0, 2, 3).contiguous()
    return img_dchw

def _affine_2d_label(y_hw, angle, dx, dy):
    # y: [H,W] or [1,H,W] -> [H,W]
    if y_hw.dim() == 3 and y_hw.shape[0] == 1:
        y_hw = y_hw[0]
    if y_hw.dim() != 2:
        raise ValueError(f"Label slice must be [H,W], got {tuple(y_hw.shape)}")

    y1 = y_hw.unsqueeze(0).float()  # [1,H,W]
    y1 = TF.affine(
        y1, angle=angle, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST
    )
    return y1[0].long()

# -------- on-set ops (A-scheme: same chosen set S for all sub-perturbations) --------

def _jitter_on_set_fixed(img_dchw, lab_dhw, rng, chosen, tmax, rmax):
    D = img_dchw.shape[0]
    out_img, out_lab = [], []
    for d in range(D):
        if d in chosen:
            dx = int(rng.randint(-tmax, tmax + 1))
            dy = int(rng.randint(-tmax, tmax + 1))
            angle = float(rng.uniform(-rmax, rmax))
            img_d = TF.affine(
                img_dchw[d], angle=angle, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR
            )
            lab_d = _affine_2d_label(lab_dhw[d], angle, dx, dy)
        else:
            img_d = img_dchw[d]
            lab_d = lab_dhw[d]
        out_img.append(img_d)
        out_lab.append(lab_d)
    return torch.stack(out_img, 0), torch.stack(out_lab, 0)

def _shuffle_on_set(img_dchw, lab_dhw, rng, chosen):
    idx = sorted(list(chosen))
    if len(idx) <= 1:
        return img_dchw, lab_dhw
    perm = idx.copy()
    rng.shuffle(perm)

    out_img = img_dchw.clone()
    out_lab = lab_dhw.clone()
    for src, dst in zip(idx, perm):
        out_img[src] = img_dchw[dst]
        out_lab[src] = lab_dhw[dst]
    return out_img, out_lab

def _time_shift_on_set_fixed(img_dchw, lab_dhw, rng, chosen, smax):
    D = img_dchw.shape[0]
    out_img = img_dchw.clone()
    out_lab = lab_dhw.clone()
    for d in chosen:
        dt = int(rng.randint(-smax, smax + 1))
        dd = min(max(d + dt, 0), D - 1)
        out_img[d] = img_dchw[dd]
        out_lab[d] = lab_dhw[dd]
    return out_img, out_lab

def _drop_duplicate_on_set_fixed(img, lab, rng, chosen):
    """
    NOTE: Here we apply drop/dup only on chosen slices, but resampling back to D
    can still slightly affect global ordering. This mirrors your classification strict impl.
    """
    D = img.shape[0]
    S = sorted(list(chosen))
    if len(S) == 0:
        return img, lab

    drop_count = len(S) // 2
    dup_count = len(S) - drop_count
    drop_idx = set(rng.choice(S, size=drop_count, replace=False).tolist()) if drop_count > 0 else set()

    img_list, lab_list = [], []
    for d in range(D):
        if d not in chosen:
            img_list.append(img[d])
            lab_list.append(lab[d])
            continue
        if d in drop_idx:
            continue
        img_list.append(img[d])
        lab_list.append(lab[d])

    kept = [d for d in S if d not in drop_idx]
    if len(kept) == 0:
        kept = [S[int(rng.randint(0, len(S)))]]
        img_list.append(img[kept[0]])
        lab_list.append(lab[kept[0]])

    dup_src = rng.choice(kept, size=dup_count, replace=True).tolist() if dup_count > 0 else []
    for d in dup_src:
        img_list.append(img[d])
        lab_list.append(lab[d])

    imgk = torch.stack(img_list, 0)   # [D',C,H,W]
    labk = torch.stack(lab_list, 0)   # [D',H,W]
    ridx = torch.linspace(0, imgk.shape[0] - 1, steps=D).round().long()
    return imgk[ridx], labk[ridx]

def _temporal_noise_strict(img_dchw, lab_dhw, rng, ratio, cfg):
    D = img_dchw.shape[0]
    chosen = _choose_indices(D, rng, ratio)
    img1, lab1 = _time_shift_on_set_fixed(img_dchw, lab_dhw, rng, chosen, smax=cfg["smax"])
    img2, lab2 = _drop_duplicate_on_set_fixed(img1, lab1, rng, chosen)
    return img2, lab2

def _combo_motion_strict(img_dchw, lab_dhw, rng, ratio, cfg):
    D = img_dchw.shape[0]
    chosen = _choose_indices(D, rng, ratio)

    img1, lab1 = _jitter_on_set_fixed(img_dchw, lab_dhw, rng, chosen, tmax=cfg["tmax"], rmax=cfg["rmax"])
    img2, lab2 = _shuffle_on_set(img1, lab1, rng, chosen)
    img3, lab3 = _time_shift_on_set_fixed(img2, lab2, rng, chosen, smax=cfg["smax"])
    img4, lab4 = _drop_duplicate_on_set_fixed(img3, lab3, rng, chosen)
    return img4, lab4

def apply_motion_perturb_seg(img, label, perturb_name="clean", base_seed=0, step_idx=0):
    """
    img: usually [C,D,H,W] or [D,C,H,W]; label: [D,H,W] or [H,W] (or [1,D,H,W] depending on dataset)
    Only used in evaluation/testing.
    """
    if perturb_name == "clean":
        return img, label

    rng = np.random.RandomState(base_seed * 1000003 + step_idx)

    # normalize label to [D,H,W]
    if label.dim() == 4 and label.shape[0] == 1:
        label = label[0]
    if label.dim() == 2:
        # make it [D,H,W] by repeating (rare)
        # will later be checked against D
        pass

    img_dchw, layout = _to_DCHW(img)
    if layout == "CHW":
        # 2D case: for your project this is unlikely; keep as clean or raise
        raise ValueError("Got 2D [C,H,W] input for segmentation perturbation; expected 3D with depth D.")
    D = img_dchw.shape[0]

    if label.dim() == 2:
        label = label.unsqueeze(0).repeat(D, 1, 1)
    if label.dim() != 3 or label.shape[0] != D:
        raise ValueError(f"Depth mismatch: img D={D}, label shape={tuple(label.shape)}")

    ratio = _parse_ratio_from_name(perturb_name)
    cfg = _get_severity_cfg(ratio)

    if perturb_name.startswith("jitter_p"):
        chosen = _choose_indices(D, rng, ratio)
        img_dchw, label = _jitter_on_set_fixed(img_dchw, label, rng, chosen, cfg["tmax"], cfg["rmax"])
    elif perturb_name.startswith("shuffle_p"):
        chosen = _choose_indices(D, rng, ratio)
        img_dchw, label = _shuffle_on_set(img_dchw, label, rng, chosen)
    elif perturb_name.startswith("temporal_p"):
        img_dchw, label = _temporal_noise_strict(img_dchw, label, rng, ratio, cfg)
    elif perturb_name.startswith("combo_p"):
        img_dchw, label = _combo_motion_strict(img_dchw, label, rng, ratio, cfg)
    else:
        raise ValueError(f"Unknown perturb_name: {perturb_name}")

    img_out = _from_DCHW(img_dchw, layout)
    return img_out, label


# =========================
#  Metrics (keep your original)
# =========================

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

def eval_dice(pred_y, gt_y, classes=2):
    pred_y = torch.argmax(torch.softmax(pred_y, dim=1), dim=1)
    all_dice, all_hd95 = [], []

    pred_y = pred_y.cpu().detach().numpy()
    gt_y = gt_y.cpu().detach().numpy()

    for cls in range(1, classes):
        dice, hd95 = calculate_metric_percase(pred_y == cls, gt_y == cls)
        all_dice.append(dice)
        all_hd95.append(hd95)

    all_dice = torch.tensor(all_dice).cuda()
    all_hd95 = torch.tensor(all_hd95).cuda()
    return torch.mean(all_dice), torch.mean(all_hd95)


@torch.no_grad()
def eval_seg(model, val_loader, device, classes=2, perturb_name="clean", perturb_seed=0):
    all_dice, all_hd95 = [], []
    model.eval()

    for i, batch in enumerate(val_loader):
        # IMPORTANT: avoid squeeze() which breaks shapes; batch_size=1
        img = batch["image"][0]    # tensor on CPU
        label = batch["label"][0]  # tensor on CPU

        # apply perturb on CPU
        if perturb_name != "clean":
            img, label = apply_motion_perturb_seg(img, label, perturb_name=perturb_name,
                                                  base_seed=perturb_seed, step_idx=i)

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        # keep same feeding logic: [D,C,H,W] -> [1,C,D,H,W]
        # if img is [C,D,H,W], this permute works; if [D,C,H,W] it becomes wrong.
        # In your project, it is usually [D,C,H,W] before permute(0,2,1,3,4) in training.
        # Here we enforce [D,C,H,W] -> [1,C,D,H,W]
        img_dchw, layout = _to_DCHW(img)  # layout will be DCHW or CDHW on GPU too
        if layout == "CDHW":
            # [C,D,H,W] -> [D,C,H,W]
            img_dchw = img_dchw
        elif layout == "DCHW":
            pass
        else:
            raise ValueError(f"Unexpected layout in eval: {layout}")

        img_in = img_dchw.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1,C,D,H,W]

        output = model(img_in)
        dice, hd95 = eval_dice(output, label, classes=classes)
        all_dice.append(dice.item())
        all_hd95.append(hd95.item())

    return float(np.mean(all_dice)), float(np.mean(all_hd95))


def _parse_type_ratio(name: str):
    t, p = name.rsplit("_p", 1)
    return t, int(p)

def motion_bias_diagnostics_seg(test_loader, model, args, device="cuda", classes=2):
    ratios = [20, 50, 80]
    types = ["jitter", "shuffle", "temporal", "combo"]
    suite = [f"{t}_p{p}" for t in types for p in ratios]  # 12

    seeds = [0]  # or [0,1,2]

    clean_dice, clean_hd95 = eval_seg(model, test_loader, device, classes=classes,
                                      perturb_name="clean", perturb_seed=0)

    rows = []
    for name in suite:
        t, p = _parse_type_ratio(name)
        dice_list, hd_list = [], []
        for sd in seeds:
            d, h = eval_seg(model, test_loader, device, classes=classes,
                            perturb_name=name, perturb_seed=sd)
            dice_list.append(d)
            hd_list.append(h)

        d_mean, d_std = float(np.mean(dice_list)), float(np.std(dice_list))
        h_mean, h_std = float(np.mean(hd_list)), float(np.std(hd_list))
        drop = (clean_dice - d_mean) / clean_dice if clean_dice > 1e-9 else 0.0
        rows.append((t, p, d_mean, d_std, h_mean, h_std, drop))

    print("\n=== Motion-bias diagnostics (Segmentation, 12 tests, severity by ratio p%) ===")
    print(f"clean                 Dice {clean_dice*100:.1f}  HD95 {clean_hd95:.2f}")
    for (t, p, dm, ds, hm, hs, drop) in rows:
        print(f"{t:8s} p={p:>3d}%   Dice {dm*100:.1f}±{ds*100:.1f}  HD95 {hm:.2f}±{hs:.2f}  DiceDrop {drop*100:.2f}%")

    return {"clean_dice": clean_dice, "clean_hd95": clean_hd95, "rows": rows}




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


# def eval(model, val_loader, device, classes):
#     all_dice = []
#     all_hd95 = []
#     model.eval()

#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):
#             img, label = batch['image'].to(device).squeeze(), batch['label'].to(device).squeeze()
#             img = img.unsqueeze(0).permute(0, 2, 1, 3, 4)
#             output = model(img)

#             dice, hd95 = eval_dice(output, label, classes=classes)
#             all_dice.append(dice.item())
#             all_hd95.append(hd95.item())

#     return np.mean(np.array(all_dice)), np.mean(np.array(all_hd95))
def eval(model, val_loader, device, classes, perturb_name="clean", perturb_seed=0):
    all_dice = []
    all_hd95 = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # 先在 CPU 上取出
            img = batch['image'][0]   # 保留原始维度，不乱 squeeze
            label = batch['label'][0]

            # 扰动只在评测用
            img, label = apply_perturb_seg_batch(img, label, perturb_name=perturb_name,
                                                 seed=perturb_seed, step_idx=i)

            img = img.to(device)
            label = label.to(device)

            # 维度保持你原来的逻辑（适配你工程：输入是 [D,C,H,W] -> [1,C,D,H,W]）
            # 如果你的 img 本来就是 [D,C,H,W]，这句会变成 [1,C,D,H,W]
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

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of test set is: {}".format(len(db_test)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn,collate_fn=dynamic_padding_collate_fn)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    # if args.test:
    #     model.eval()
    #     model.load_state_dict(torch.load(args.pretrained_model_weights, map_location='cpu'))
    #     test_dice, test_hd95 = eval(model, testloader, 'cuda', classes=2)
    #     print('Test Dice: %.1f, HD95: %.1f' % (test_dice * 100., test_hd95))
    #     exit(0)
    if args.test:
        model.eval()
        model.load_state_dict(torch.load(args.pretrained_model_weights, map_location="cpu"))
        model.cuda()
        device = "cuda"

        # 强烈建议：num_workers=0，避免你之前的 segfault
        # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        motion_bias_diagnostics_seg(testloader, model, args, device=device, classes=2)
        exit(0)



    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=5e-2)

    if args.wandb:
        wandb.init(project='synapse-segmentation', config=args, dir="/mnt/tqy/wandb/")
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
