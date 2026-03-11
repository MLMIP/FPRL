import argparse
import json
import os
from cv2 import line
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from torch import nn
from tqdm import tqdm
import numpy as np
import random

from sklearn.metrics import f1_score

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from timm.models import create_model
from datasets import UCF101, HMDB51, Kinetics
from models import get_vit_base_patch16_224, get_aux_token_vit, SwinTransformer3D
from video_sm.models.videomamba import videomamba_small
from models.endomamba_classification import endomamba_small
from video_sm.models.videomae_v2 import vit_small_patch16_224
from utils import utils
from utils.meters import TestMeter
from utils.parser import load_config
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

RATIOS = [0.2, 0.5, 0.8]  # 20% / 50% / 80%
# 统一的比例与固定幅度表（方案A：比例=全局受扰帧比例）
# SEVERITY_TABLE = {
#     0.2: {"tmax": 2,  "rmax": 1.0, "smax": 1},  # p=20%
#     0.5: {"tmax": 6,  "rmax": 3.0, "smax": 2},  # p=50%
#     0.8: {"tmax": 12, "rmax": 6.0, "smax": 4},  # p=80%
# }
SEVERITY_TABLE = {
  0.2: {"tmax": 8,  "rmax": 5.0,  "smax": 3},   # p=20%
  0.5: {"tmax": 16, "rmax": 10.0, "smax": 6},   # p=50%
  0.8: {"tmax": 32, "rmax": 15.0, "smax": 10},  # p=80%
}


def _parse_ratio_from_name(name: str) -> float:
    # e.g. "temporal_p20" -> 0.2
    p_str = name.rsplit("_p", 1)[1]
    return float(p_str) / 100.0

def _get_severity_cfg(ratio: float):
    # ratio 可能会是 0.2000000001，做个 round
    r = round(float(ratio), 1)
    if r not in SEVERITY_TABLE:
        raise ValueError(f"Unsupported ratio={ratio}. Supported: {list(SEVERITY_TABLE.keys())}")
    return SEVERITY_TABLE[r]


def _to_TCHW(video):
    """
    Accept video tensor in [C,T,H,W] or [T,C,H,W], return [T,C,H,W] and a flag.
    """
    if video.dim() != 4:
        raise ValueError(f"Expect 4D video, got {video.shape}")
    # guess layout by channel dim
    if video.shape[0] in [1, 3]:  # [C,T,H,W]
        return video.permute(1, 0, 2, 3).contiguous(), "CTHW"
    else:  # [T,C,H,W]
        return video, "TCHW"

def _back_from_TCHW(video_tchw, layout_flag):
    if layout_flag == "CTHW":
        return video_tchw.permute(1, 0, 2, 3).contiguous()
    return video_tchw

def _choose_indices(T, rng, ratio):
    k = int(round(T * float(ratio)))
    k = max(0, min(T, k))
    if k == 0:
        return set()
    return set(rng.choice(T, size=k, replace=False).tolist())

def _camera_jitter_on_set_fixed(video, rng, chosen, tmax, rmax):
    v, flag = _to_TCHW(video)
    T = v.shape[0]
    out = []
    for t in range(T):
        frame = v[t]
        if t in chosen:
            dx = int(rng.randint(-tmax, tmax + 1))
            dy = int(rng.randint(-tmax, tmax + 1))
            angle = float(rng.uniform(-rmax, rmax))
            frame = TF.affine(
                frame, angle=angle, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR
            )
        out.append(frame)
    out = torch.stack(out, dim=0)
    return _back_from_TCHW(out, flag)


def _partial_shuffle_on_set(video, rng, chosen):
    v, flag = _to_TCHW(video)
    idx = sorted(list(chosen))
    if len(idx) <= 1:
        return video
    perm = idx.copy()
    rng.shuffle(perm)
    out = v.clone()
    for src, dst in zip(idx, perm):
        out[src] = v[dst]
    return _back_from_TCHW(out, flag)

def _time_shift_on_set_fixed(video, rng, chosen, smax):
    v, flag = _to_TCHW(video)
    T = v.shape[0]
    out = []
    for t in range(T):
        if t in chosen:
            dt = int(rng.randint(-smax, smax + 1))
            tt = min(max(t + dt, 0), T - 1)
            out.append(v[tt])
        else:
            out.append(v[t])
    out = torch.stack(out, dim=0)
    return _back_from_TCHW(out, flag)


def _drop_duplicate_on_set_fixed(video, rng, chosen):
    """
    只在 chosen 上做 drop/dup，且数量固定：
    - drop_count = floor(|S|/2)
    - dup_count  = |S| - drop_count
    最后 resample 回原长度 T
    """
    v, flag = _to_TCHW(video)
    T = v.shape[0]
    S = sorted(list(chosen))
    if len(S) == 0:
        return video

    drop_count = len(S) // 2
    dup_count = len(S) - drop_count

    drop_idx = set(rng.choice(S, size=drop_count, replace=False).tolist()) if drop_count > 0 else set()

    frames = []
    for t in range(T):
        if t not in chosen:
            frames.append(v[t])
            continue
        if t in drop_idx:
            continue
        frames.append(v[t])

    # 在保留下来的 chosen 帧里 duplicate 固定数量
    chosen_kept = [t for t in S if t not in drop_idx]
    if len(chosen_kept) == 0:
        chosen_kept = [S[int(rng.randint(0, len(S)))]]
        frames.append(v[chosen_kept[0]])

    dup_src = rng.choice(chosen_kept, size=dup_count, replace=True).tolist() if dup_count > 0 else []
    for t in dup_src:
        frames.append(v[t])

    frames = torch.stack(frames, dim=0)
    idx = torch.linspace(0, frames.shape[0] - 1, steps=T).round().long()
    out = frames[idx]
    return _back_from_TCHW(out, flag)

def _temporal_noise_strict(video, rng, ratio, cfg):
    v, _ = _to_TCHW(video)
    T = v.shape[0]
    chosen = _choose_indices(T, rng, ratio)

    # 同一集合 chosen 上叠加 shift + drop/dup
    video = _time_shift_on_set_fixed(video, rng, chosen, smax=cfg["smax"])
    video = _drop_duplicate_on_set_fixed(video, rng, chosen)
    return video

def _combo_motion_strict(video, rng, ratio, cfg):
    v, _ = _to_TCHW(video)
    T = v.shape[0]
    chosen = _choose_indices(T, rng, ratio)

    video = _camera_jitter_on_set_fixed(video, rng, chosen, tmax=cfg["tmax"], rmax=cfg["rmax"])
    video = _partial_shuffle_on_set(video, rng, chosen)
    video = _time_shift_on_set_fixed(video, rng, chosen, smax=cfg["smax"])
    video = _drop_duplicate_on_set_fixed(video, rng, chosen)
    return video


def _parse_ratio_from_name(name: str) -> float:
    # e.g., "temporal_p20" -> 0.2
    try:
        p_str = name.rsplit("_p", 1)[1]  # only split at the last "_p"
        return float(p_str) / 100.0
    except Exception as e:
        raise ValueError(f"Bad perturb_name format: {name}. Expect like 'jitter_p20'") from e


def apply_motion_perturb_batch(inp, sample_idx, base_seed, perturb_name):
    """
    inp: [B,C,T,H,W] or [B,T,C,H,W] or [B,N,C,T,H,W] (multi-crop)
    sample_idx: tensor/list length B
    """
    if perturb_name == "clean":
        return inp

    # handle multi-crop: [B,N,...]
    if inp.dim() == 6:
        B, N = inp.shape[0], inp.shape[1]
        out = inp.clone()
        for b in range(B):
            sid = int(sample_idx[b])
            rng = np.random.RandomState(base_seed * 1000003 + sid)
            for n in range(N):
                out[b, n] = apply_motion_perturb_batch(out[b, n].unsqueeze(0), [sid], base_seed, perturb_name)[0]
        return out

    if inp.dim() != 5:
        raise ValueError(f"Expect 5D or 6D input, got {inp.shape}")

    B = inp.shape[0]
    out = inp.clone()

    for b in range(B):
        sid = int(sample_idx[b])
        rng = np.random.RandomState(base_seed * 1000003 + sid)

        video = out[b]  # [C,T,H,W] or [T,C,H,W]

        if perturb_name == "clean":
            pass

        elif perturb_name.startswith("jitter_p"):
            ratio = _parse_ratio_from_name(perturb_name)
            cfg = _get_severity_cfg(ratio)
            v, _ = _to_TCHW(video)
            chosen = _choose_indices(v.shape[0], rng, ratio)
            video = _camera_jitter_on_set_fixed(video, rng, chosen, tmax=cfg["tmax"], rmax=cfg["rmax"])

        elif perturb_name.startswith("shuffle_p"):
            ratio = _parse_ratio_from_name(perturb_name)
            v, _ = _to_TCHW(video)
            chosen = _choose_indices(v.shape[0], rng, ratio)
            video = _partial_shuffle_on_set(video, rng, chosen)

        elif perturb_name.startswith("temporal_p"):
            ratio = _parse_ratio_from_name(perturb_name)
            cfg = _get_severity_cfg(ratio)
            video = _temporal_noise_strict(video, rng, ratio, cfg)

        elif perturb_name.startswith("combo_p"):
            ratio = _parse_ratio_from_name(perturb_name)
            cfg = _get_severity_cfg(ratio)
            video = _combo_motion_strict(video, rng, ratio, cfg)

        else:
            raise ValueError(f"Unknown perturb_name: {perturb_name}")

        out[b] = video


    return out
def _parse_type_ratio(name: str):
    # e.g. "temporal_p20" -> ("temporal", 20)
    t, p = name.rsplit("_p", 1)
    return t, int(p)

def motion_bias_diagnostics(val_loader, model, linear_classifier, args):
    ratios = [20, 50, 80]
    types = ["jitter", "shuffle", "temporal", "combo"]
    suite = [f"{t}_p{p}" for t in types for p in ratios]  # 12个，顺序固定

    seeds = [0]  # 或 [0,1,2]

    clean_stats, clean_f1 = validate_network(
        val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens,
        perturb_name="clean", perturb_seed=0
    )

    rows = []
    for name in suite:
        t, p = _parse_type_ratio(name)

        f1_list, acc_list = [], []
        for sd in seeds:
            stats, f1 = validate_network(
                val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens,
                perturb_name=name, perturb_seed=sd
            )
            f1_list.append(f1)
            acc_list.append(stats["acc1"])

        f1_mean, f1_std = float(np.mean(f1_list)), float(np.std(f1_list))
        acc_mean, acc_std = float(np.mean(acc_list)), float(np.std(acc_list))
        drop = (clean_f1 - f1_mean) / clean_f1 if clean_f1 > 1e-9 else 0.0

        rows.append((t, p, acc_mean, acc_std, f1_mean, f1_std, drop))

    if utils.is_main_process():
        print("\n=== Motion-bias diagnostics (12 tests, severity by ratio p%) ===")
        print(f"clean                 Acc1 {clean_stats['acc1']:.2f}  F1 {clean_f1:.4f}")
        for (t, p, acc_m, acc_s, f1_m, f1_s, drop) in rows:
            print(f"{t:8s} p={p:>3d}%   Acc1 {acc_m:.2f}±{acc_s:.2f}  F1 {f1_m:.4f}±{f1_s:.4f}  Drop {drop*100:.2f}%")

    return {"clean_f1": clean_f1, "clean_acc1": float(clean_stats["acc1"]), "rows": rows}



def _motion_profile(video, rng, level: str):
    """
    Severity is quantified ONLY by ratio p.
    Parameter amplitudes are randomized per-video (but reproducible via rng).
    """
    p_map = {"s": 0.2, "m": 0.5, "l": 0.8}
    p = p_map[level]

    # 关键：避免“叠加太多导致>p 的破坏”
    # 这里采用“同一视频随机挑一种时序扰动 + 一种空间扰动”，都用同一个比例 p。
    # 这样你写 protocol 会更干净：p% frames are corrupted.
    # (如果你坚持四种全叠加，我也能给，但那样实际破坏比例会远大于 p)
    video = _camera_jitter_ratio_random(
        video, rng, ratio=p,
        translate_max_range=(1, 12),   # 幅度随机（每个视频采样）
        rotate_max_range=(0.5, 6.0)
    )

    # 从三种“时序类”里随机选一种（参数幅度同样随机），但比例固定为 p
    op = int(rng.randint(0, 3))
    if op == 0:
        video = _partial_shuffle(video, rng, ratio=p)
    elif op == 1:
        video = _time_shift_ratio_random(video, rng, ratio=p, shift_max_range=(1, 4))
    else:
        video = _drop_duplicate_exact(video, rng, ratio=p)

    return video




def eval_finetune(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    args.output_dir = args.output_dir + args.arch + '_s' + str(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(f"{args.output_dir}/config.json", "w"), indent=4)

    # ============ preparing data ... ============ 
    config = load_config(args)
    config.TEST.NUM_SPATIAL_CROPS = 1
    config.DATA.PATH_TO_DATA_DIR = args.data_path + 'splits'
    config.DATA.PATH_PREFIX = args.data_path + 'videos'
    config.DATA.USE_FLOW = False
    if args.dataset == "ucf101":
        dataset_train = UCF101(cfg=config, mode="train", num_retries=10)
        dataset_val = UCF101(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "hmdb51":
        dataset_train = HMDB51(cfg=config, mode="train", num_retries=10)
        dataset_val = HMDB51(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    elif args.dataset == "kinetics400":
        dataset_train = Kinetics(cfg=config, mode="train", num_retries=10)
        dataset_val = Kinetics(cfg=config, mode="val", num_retries=10)
        config.TEST.NUM_SPATIAL_CROPS = 3
    else:
        raise NotImplementedError(f"invalid dataset: {args.dataset}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============ 
    if config.DATA.USE_FLOW or config.MODEL.TWO_TOKEN:
        model = get_aux_token_vit(cfg=config, no_head=True)
        model_embed_dim = 2 * model.embed_dim
    else:
        if args.arch == "vit_base":
            model = get_vit_base_patch16_224(cfg=config, no_head=True)
            model_embed_dim = model.embed_dim
        elif args.arch == "swin":
            model = SwinTransformer3D(depths=[2, 2, 18, 2], embed_dim=128, num_heads=[4, 8, 16, 32])
            model_embed_dim = 1024
        # elif args.arch == "endomamba":
        #     model = get_endomamba_small(cfg=config, no_head=False, pretrained=not args.scratch, 
        #                                 num_classes=args.num_labels)  
        #     model_embed_dim = 384  # Set the embed dimension to 384 as per endomamba configuration
        else:
            if args.arch == "videomaev2":
                args.arch = "vit_small_patch16_224"
            model = create_model(args.arch, no_head=False, pretrained=not args.scratch, num_classes=args.num_labels)
            model_embed_dim = model.embed_dim  # Set the embed dimension to 384 as per endomamba configuration
        # else:
        #     raise Exception(f"invalid model: {args.arch}")

    model.cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    if utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")

    # ============ Remove Linear Classifier for "endomamba" case ... ============ 
    if "endomamba" in args.arch or "videomamba" in args.arch or "vit_small_patch16_224" in args.arch:
        linear_classifier = None  # Directly set to None for endomamba since it includes its own head
    else:
        linear_classifier = LinearClassifier(model_embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens)),
                                             num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # If test mode, load weights and perform testing
    if args.test:
        model.eval()
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        print('best_f1', state_dict["best_f1"])
        model.load_state_dict(state_dict["backbone_state_dict"])
        test_stats, f1 = validate_network(val_loader, model, linear_classifier, args.n_last_blocks,
                                          args.avgpool_patchtokens)
        print(f"F1 score of the network on the {len(dataset_val)} test images: {f1 * 100:.1f}%")
        exit(0)

    scaled_lr = args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.

    # set optimizer
    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'lr': scaled_lr}],
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    if linear_classifier is not None:
        optimizer.add_param_group(
            {'params': linear_classifier.parameters(), 'lr': scaled_lr}
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0., "best_f1": 0.}
    # utils.restart_from_checkpoint(
    #     os.path.join(args.output_dir, "checkpoint.pth.tar"),
    #     run_variables=to_restore,
    #     state_dict=linear_classifier,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    # )
    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(args, model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # 1) clean 验证（用于 best_f1 选择 & checkpoint 保存）
            test_stats, f1 = validate_network(
                val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens,
                perturb_name="clean", perturb_seed=args.seed
            )
            print(f"F1 score at epoch {epoch} on clean: {f1 * 100:.1f}%")

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         "test_f1_clean": f1}

            # 2) 额外跑一遍运动扰动诊断（只打印/记录，不参与 best 选择）
            diag = motion_bias_diagnostics(val_loader, model, linear_classifier, args)

            # 可选：把诊断摘要写进 log_stats（避免 log 太长，建议只写平均 drop）
            mean_drop = float(np.mean([r[-1] for r in diag["rows"]])) if len(diag["rows"]) else 0.0
            log_stats["motion_mean_drop"] = mean_drop
            log_stats["motion_clean_f1"] = diag["clean_f1"]

            # 3) best checkpoint 逻辑仍然只看 clean f1
            if f1 > best_f1 and utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                best_f1 = max(best_f1, f1)
                save_dict = {
                    "epoch": epoch + 1,
                    "backbone_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": best_f1,
                }
                if linear_classifier is not None:
                    save_dict["linear_classifier"] = linear_classifier.state_dict()
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

            best_f1 = max(best_f1, f1)
            print(f'Max clean F1 so far: {best_f1 * 100:.1f}%')
#         if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
#             test_stats, f1 = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
#             print(f"F1 score at epoch {epoch} of the network on the {len(dataset_val)} test images: {f1 * 100:.1f}%")
#             log_stats = {**{k: v for k, v in log_stats.items()},
#                          **{f'test_{k}': v for k, v in test_stats.items()}}

#             if f1 > best_f1 and utils.is_main_process():
#                 with (Path(args.output_dir) / "log.txt").open("a") as f:
#                     f.write(json.dumps(log_stats) + "\n")
#                 best_f1 = max(best_f1, f1)
#                 save_dict = {
#                     "epoch": epoch + 1,
#                     "backbone_state_dict": model.state_dict(),
#                     "optimizer": optimizer.state_dict(),
#                     "scheduler": scheduler.state_dict(),
#                     "best_f1": best_f1,
#                 }
#                 if linear_classifier is not None:
#                     save_dict["linear_classifier"] = linear_classifier.state_dict()
#                 torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

#             best_f1 = max(best_f1, f1)
#             print(f'Max F1 score so far: {best_f1 * 100:.1f}%')

# Training function
def train(args, model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    model.train()
    if linear_classifier is not None:
        linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, sample_idx, meta) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(inp)
        if linear_classifier is not None:
            output = linear_classifier(output)

        # print("Output shape:", output.shape)
        # print("Target shape:", target.shape)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# Validation function
@torch.no_grad()
# def validate_network(val_loader, model, linear_classifier, n, avgpool):
def validate_network(val_loader, model, linear_classifier, n, avgpool, perturb_name="clean", perturb_seed=0):
    model.eval()
    if linear_classifier is not None:
        linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    all_target = []
    all_output = []
    for (inp, target, sample_idx, meta) in metric_logger.log_every(val_loader, 20, header):
        # 只在验证/测试时做运动扰动（train() 完全不受影响）
        inp = apply_motion_perturb_batch(inp, sample_idx, base_seed=perturb_seed, perturb_name=perturb_name)
        
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(inp)
        if linear_classifier is not None:
            output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, = utils.accuracy(output, target, topk=(1,))
        all_target.extend(target.detach().cpu().numpy())
        all_output.extend(np.argmax(output.detach().cpu().numpy(), axis=1))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    f1 = f1_score(all_target, all_output)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, f1


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='endomamba_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--lc_pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/data/tqy/PolypDiag/', type=str)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/data/tqy/out/Classification/MIX12", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--dataset', default="ucf101", help='Dataset: ucf101 / hmdb51')
    parser.add_argument('--use_flow', default=False, type=utils.bool_flag, help="use flow teacher")

    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--pretrained_model_weights', default='', type=str, help='pre-trained weights')
    parser.add_argument('--seed', type=int,
                        default=3, help='random seed')
    
    # config file
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", type=str,
                        default="./models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml")
    parser.add_argument("--opts", help="See utils/defaults.py for all options", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    eval_finetune(args)
