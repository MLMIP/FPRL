import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models.videomamba import *
from timm.models import create_model

# ========== 热力图保存 ==========
def generate_heatmap(feature, save_path, title=None, enhance_low_values=True):
    """
    feature: numpy array [H, W]
    save_path: 保存路径
    title: 图像标题
    enhance_low_values: 是否增强低值区域的对比度
    """
    # ====== 数值处理部分 ======
    if enhance_low_values:
        # 1) 对数压缩
        feature = np.log1p(feature)

        # 2) 归一化到 [0,1]
        f_min, f_max = feature.min(), feature.max()
        if f_max > f_min:
            feature = (feature - f_min) / (f_max - f_min + 1e-8)

        # 3) 分位数阈值（这里是 70%，即底 70% 全黑，顶 30% 上色）
        thr = np.percentile(feature, 70)

        # 4) 构造映射：低 70% -> 0，高 30% -> [0.3, 1.0]
        mapped = np.zeros_like(feature)
        low_mask = feature <= thr
        high_mask = ~low_mask

        mapped[low_mask] = 0.0
        mapped[high_mask] = 0.3 + (feature[high_mask] - thr) / (1.0 - thr + 1e-8) * 0.7

        data = mapped        # ★ 之后只用 data 画图
        vmin, vmax = 0.0, 1.0
    else:
        data = feature       # 不增强，直接用原始 feature
        vmin, vmax = None, None

    # ====== 画图 + 无白边保存部分 ======
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(frameon=False)          # 不画外框
    fig.set_size_inches(4, 4)

    ax = plt.Axes(fig, [0., 0., 1., 1.])    # 占满画布
    fig.add_axes(ax)
    ax.set_axis_off()

    ax.imshow(data, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

    # 如需标题可以暂时打开，但会产生边距
    # if title is not None:
    #     ax.set_title(title)

    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ========== 图像预处理（给模型用） ==========

def preprocess_image(image_path):
    """加载并预处理图像 -> model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # [1, 3, H, W]

# ========== 简单 resize 保存原图（给可视化用） ==========

def save_resized_original(image_path, save_path, size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)

# ========== patch map 上采样到图像尺寸 ==========

def upsample_patch_map(patch_map, original_shape=(224, 224)):
    """
    patch_map: [Gh, Gw] 的 patch 级热力图
    返回: [H, W] 的像素级热力图
    """
    grid_h, grid_w = patch_map.shape
    H, W = original_shape
    patch_h, patch_w = H // grid_h, W // grid_w
    heatmap_resized = np.zeros((H, W), dtype=np.float32)
    
    for i in range(grid_h):
        for j in range(grid_w):
            h_start, h_end = i * patch_h, (i + 1) * patch_h
            w_start, w_end = j * patch_w, (j + 1) * patch_w
            heatmap_resized[h_start:h_end, w_start:w_end] = patch_map[i, j]
    return heatmap_resized

# ========== 从 patch 特征计算热力图 ==========

def compute_heatmap_from_patches(
    features, 
    original_shape=(224, 224), 
    patch_size=16, 
    enhance_low_values=True,
    return_patch_map=False
):
    """
    features: torch.Tensor [B, N, M] - 单帧的 patch 特征
    original_shape: 原始图像尺寸 (H, W)
    返回:
        - 只要像素级热力图: np.array [H, W]
        - 如 return_patch_map=True: (heatmap_resized, patch_map) 
          patch_map 为 [Gh, Gw] 的 patch 级热力图
    """
    # 1. L1 范数
    patch_heat = torch.norm(features, p=1, dim=-1)  # [B, N]
    B, N = patch_heat.shape

    # 2. reshape 成 2D patch 网格
    grid_size = int(np.sqrt(N))
    patch_map = patch_heat.view(B, grid_size, grid_size)[0].detach().cpu().numpy()  # [Gh, Gw]

    # 3. 归一化 + 低值增强
    h_min, h_max = patch_map.min(), patch_map.max()
    if h_max > h_min:
        patch_map = (patch_map - h_min) / (h_max - h_min)
    if enhance_low_values:
        patch_map = np.power(patch_map, 0.4)  # 压缩高值，拉高低值

    # 4. 上采样到图像尺寸
    heatmap_resized = upsample_patch_map(patch_map, original_shape)

    if return_patch_map:
        return heatmap_resized, patch_map
    return heatmap_resized

# ========== 基于 patch-map 保留 10% 可视块的原图 ==========

def save_image_with_visible_patches(image_path, patch_map, keep_ratio, save_path, size=(224, 224)):
    """
    image_path: 原图路径
    patch_map: [Gh, Gw] 的综合 patch 热力图
    keep_ratio: 保留的 patch 比例（0.1 -> 10%）
    save_path: 输出路径
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    img_np = np.array(img)

    grid_h, grid_w = patch_map.shape
    num_patches = grid_h * grid_w
    keep_num = max(1, int(num_patches * keep_ratio))

    # 用综合热力图作为采样概率
    flat = patch_map.flatten().astype(np.float64)
    flat = flat - flat.min()
    if flat.sum() <= 0:
        prob = np.ones_like(flat) / len(flat)
    else:
        prob = flat / flat.sum()

    indices = np.random.choice(len(flat), size=keep_num, replace=False, p=prob)
    keep_mask = np.zeros(len(flat), dtype=bool)
    keep_mask[indices] = True

    H, W, _ = img_np.shape
    patch_h, patch_w = H // grid_h, W // grid_w

    for idx in range(len(flat)):
        if not keep_mask[idx]:
            i = idx // grid_w
            j = idx % grid_w
            h_start, h_end = i * patch_h, (i + 1) * patch_h
            w_start, w_end = j * patch_w, (j + 1) * patch_w
            img_np[h_start:h_end, w_start:w_end, :] = 0  # 其余全设为黑

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(img_np).save(save_path)

# ========== 主入口：批量处理 original/1.jpg ~ 8.jpg ==========

if __name__ == "__main__":
    np.random.seed(0)  # 方便复现随机热力图 & 10%采样

    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_model = create_model(
        "videomamba_small",
        num_frames=2,
        pretrained=True,
        with_head=False
    )
    teacher_model.to(device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    img_dir = "original"
    out_root = "mask_results"  # 所有结果保存在这个根目录下

    # 四对帧：(1,2), (3,4), (5,6), (7,8)
    for pair_idx in range(4):
        idx1 = pair_idx * 2 + 1
        idx2 = pair_idx * 2 + 2

        img1_path = os.path.join(img_dir, f"{idx1}.jpg")
        img2_path = os.path.join(img_dir, f"{idx2}.jpg")

        # 输出子目录，例如 results/pair_1/
        pair_out_dir = os.path.join(out_root, f"pair_{pair_idx + 1}")

        # 1. 保存原图（224×224）
        save_resized_original(img1_path, os.path.join(pair_out_dir, f"frame1_original.jpg"))
        save_resized_original(img2_path, os.path.join(pair_out_dir, f"frame2_original.jpg"))

        # 2. 送入模型的预处理
        frame1 = preprocess_image(img1_path).to(device)
        frame2 = preprocess_image(img2_path).to(device)

        # [B, C, T, H, W]
        frames = torch.stack([frame1, frame2], dim=2)

        with torch.no_grad():
            features = teacher_model(frames)  # [1, N_all, M]

        B, N, M = features.shape
        patches_per_frame = N // 2  # 假设输出按帧顺序拼在一起

        frame1_features = features[:, :patches_per_frame, :]
        frame2_features = features[:, patches_per_frame:, :]

        # 3. videomamba 特征热力图（像素级 + patch 级）
        heatmap1, patch_map1 = compute_heatmap_from_patches(
            frame1_features, 
            original_shape=(224, 224),
            enhance_low_values=True,
            return_patch_map=True
        )
        heatmap2, patch_map2 = compute_heatmap_from_patches(
            frame2_features, 
            original_shape=(224, 224),
            enhance_low_values=True,
            return_patch_map=True
        )

        generate_heatmap(
            heatmap1,
            os.path.join(pair_out_dir, "frame1_feature_heatmap.png"),
            # title="Frame 1 Feature Heatmap",
            enhance_low_values=True  # 已在内部增强过，这里关掉
        )
        generate_heatmap(
            heatmap2,
            os.path.join(pair_out_dir, "frame2_feature_heatmap.png"),
            # title="Frame 2 Feature Heatmap",
            enhance_low_values=True
        )

        # 4. 随机热力图（patch 级随机 -> 上采样）
        rand_map1 = np.random.rand(*patch_map1.shape).astype(np.float32)
        rand_map2 = np.random.rand(*patch_map2.shape).astype(np.float32)

        # 归一化到 [0,1]
        rand_map1 /= rand_map1.max()
        rand_map2 /= rand_map2.max()

        rand_heatmap1 = upsample_patch_map(rand_map1, original_shape=(224, 224))
        rand_heatmap2 = upsample_patch_map(rand_map2, original_shape=(224, 224))

        generate_heatmap(
            rand_heatmap1,
            os.path.join(pair_out_dir, "frame1_random_heatmap.png"),
            # title="Frame 1 Random Heatmap",
            enhance_low_values=True
        )
        generate_heatmap(
            rand_heatmap2,
            os.path.join(pair_out_dir, "frame2_random_heatmap.png"),
            # title="Frame 2 Random Heatmap",
            enhance_low_values=True
        )

        # 5. 综合热力图（以特征为主：0.7 * feature + 0.3 * random）
        alpha = 0.7
        composite_patch1 = alpha * patch_map1 + (1 - alpha) * rand_map1
        composite_patch2 = alpha * patch_map2 + (1 - alpha) * rand_map2

        # 再归一化一次比较稳
        for comp in [composite_patch1, composite_patch2]:
            cmin, cmax = comp.min(), comp.max()
            if cmax > cmin:
                comp -= cmin
                comp /= (cmax - cmin)

        composite_heatmap1 = upsample_patch_map(composite_patch1, original_shape=(224, 224))
        composite_heatmap2 = upsample_patch_map(composite_patch2, original_shape=(224, 224))

        generate_heatmap(
            composite_heatmap1,
            os.path.join(pair_out_dir, "frame1_composite_heatmap.png"),
            # title="Frame 1 Composite Heatmap",
            enhance_low_values=True
        )
        generate_heatmap(
            composite_heatmap2,
            os.path.join(pair_out_dir, "frame2_composite_heatmap.png"),
            # title="Frame 2 Composite Heatmap",
            enhance_low_values=True
        )

        # 6. 基于综合热力图采样，仅保留 10% 可视块的原图
        save_image_with_visible_patches(
            img1_path,
            composite_patch1,
            keep_ratio=0.1,
            save_path=os.path.join(pair_out_dir, "frame1_10pct_visible.jpg"),
            size=(224, 224)
        )
        save_image_with_visible_patches(
            img2_path,
            composite_patch2,
            keep_ratio=0.1,
            save_path=os.path.join(pair_out_dir, "frame2_10pct_visible.jpg"),
            size=(224, 224)
        )

        print(f"Pair {pair_idx + 1} done:", img1_path, img2_path)

    print("所有结果已保存到 ./results/ 下。")
