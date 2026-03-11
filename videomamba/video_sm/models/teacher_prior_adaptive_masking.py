import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

class MaskGenerator(nn.Module):
    """
    基于特征学习的动态掩码生成器
    """
    
    def __init__(self, num_patches=392, embed_dim=768, mask_ratio=0.9, 
                 probs_network_depth=1, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.visible_patches = int(num_patches * (1 - mask_ratio))
        self.patches_per_frame = num_patches // 2  # 每帧的块数
        self.visible_per_frame = self.visible_patches // 2  # 每帧可见的patch数
        
        self.patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=embed_dim, num_frames=2, tubelet_size=1)
        
        print(f"MaskGenerator: {num_patches} patches, {mask_ratio*100}% masking, "
              f"{self.visible_patches} visible patches, {self.patches_per_frame} patches per frame, "
              f"{self.visible_per_frame} visible per frame")
        
        # 可学习的位置编码（专门用于概率预测）
        self.pos_embed_probs = nn.Parameter(torch.zeros(1, self.patches_per_frame, embed_dim))
        
        # 概率预测网络
        self.prob_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.0, 
                drop_path=0.0, norm_layer=nn.LayerNorm, init_values=0.0
            ) for _ in range(probs_network_depth)
        ])
        
        # 输出每个patch的重要性分数
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Flatten(start_dim=1)
        )
        
#         # 添加教师特征处理相关的层
#         self.teacher_proj = nn.Linear(embed_dim, 1)  # 用于方法2

#         # 注意力机制相关的层（用于方法3）
#         self.attention_key = nn.Linear(embed_dim, embed_dim)
#         self.attention_value = nn.Linear(embed_dim, embed_dim)

        # 可学习的融合权重
        # self.raw_alpha = nn.Parameter(torch.tensor(2.197))  # 可学习的融合权重
        
        self.softmax = nn.Softmax(dim=-1)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, teacher_features=None):
        """
        前向传播生成掩码
        
        Args:
            x: 输入视频张量 [batch_size, channels, frames, height, width]
            teacher_features: 教师模型的图像块特征 [batch_size, num_patches, embed_dim]
            
        Returns:
            mask: 布尔掩码 [batch_size, num_patches], True表示被掩码
            vis_idx: 可见patch的索引 [batch_size, visible_patches]
            p_x: 概率分布 [batch_size, num_patches] 
        """
        # 首先通过patch_embed提取特征
        x_embed = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]
        batch_size = x_embed.shape[0]
        
        # 分离两帧的特征
        first_frame_features = x_embed[:, :self.patches_per_frame, :]  # [batch_size, 196, embed_dim]
        second_frame_features = x_embed[:, self.patches_per_frame:, :]  # [batch_size, 196, embed_dim]
        
        # 如果有教师特征，也分离两帧
        if teacher_features is not None:
            teacher_first_frame = teacher_features[:, :self.patches_per_frame, :]
            teacher_second_frame = teacher_features[:, self.patches_per_frame:, :]
        
        # 为两帧分别计算重要性分数
        all_probs = []
        all_vis_idx = []
        
        for frame_idx, frame_features in enumerate([first_frame_features, second_frame_features]):
            # 添加位置编码并计算重要性分数
            x_probs = frame_features + self.pos_embed_probs
            for block in self.prob_blocks:
                x_probs = block(x_probs)
            
            # 计算每个patch的分数
            logits = self.score_head(x_probs)  # [batch_size, patches_per_frame]
            logits = torch.nan_to_num(logits)  # 数值稳定性

            # 如果有教师特征，将其转换为热力图并与logits结合
            if teacher_features is not None:
                # 获取当前帧的教师特征
                teacher_frame = teacher_first_frame if frame_idx == 0 else teacher_second_frame

                # 将教师特征转换为热力图（显著性分数）
                # 基于特征范数计算重要性
                teacher_heatmap = torch.norm(teacher_frame, dim=2)  # [batch_size, patches_per_frame]

                # 使用线性投影计算重要性分数
                # teacher_heatmap = self.teacher_proj(teacher_frame).squeeze(-1)  # [batch_size, patches_per_frame]

                # 基于注意力机制计算重要性
                # teacher_heatmap = self.calculate_attention_heatmap(teacher_frame)

                # 对教师热力图和logits进行归一化
                teacher_heatmap = torch.softmax(teacher_heatmap, dim=1)
                logits = torch.softmax(logits, dim=1)

                # 将教师热力图与原始logits结合
                # 可以使用加权平均或其他融合策略
                # alpha = torch.sigmoid(self.raw_alpha)  # 始终在[0,1]范围内, 可调整的权重参数
                alpha = 0.95
                p_x_frame = alpha * logits + (1 - alpha) * teacher_heatmap

            else:
                # 如果没有教师特征，使用原始logits
                p_x_frame = self.softmax(logits)
        
            all_probs.append(p_x_frame)
            
            # 基于概率采样可见patch
            vis_idx_frame = torch.multinomial(
                p_x_frame, 
                num_samples=self.visible_per_frame, 
                replacement=False
            )  # [batch_size, visible_per_frame]
            
            # 如果是第二帧，需要加上偏移量
            if frame_idx == 1:
                vis_idx_frame = vis_idx_frame + self.patches_per_frame
            
            all_vis_idx.append(vis_idx_frame)
        
        # 合并两帧的结果
        p_x = torch.cat(all_probs, dim=1)  # [batch_size, num_patches]
        vis_idx = torch.cat(all_vis_idx, dim=1)  # [batch_size, visible_patches]
        
        # 创建掩码
        mask = torch.ones(batch_size, self.num_patches, device=x.device)
        mask.scatter_(dim=1, index=vis_idx.long(), value=0.0)
        mask = mask.to(torch.bool)
        
        return p_x, vis_idx, mask
    
    def get_visibility_stats(self, x, num_samples=1000):
        """
        分析掩码模式，统计哪些patch更常被保留
        用于监控训练过程
        """
        visibility_count = torch.zeros(self.num_patches, device=x.device)
        
        with torch.no_grad():
            for _ in range(num_samples):
                _, _, mask = self.forward(x)
                visibility_count += (~mask).float().sum(dim=0)
        
        visibility_freq = visibility_count / (num_samples * x.shape[0])
        return visibility_freq

    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x
    
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def calculate_attention_heatmap(self, teacher_features):
    """
    基于自注意力机制计算教师特征的热力图
    """
    batch_size, num_patches, embed_dim = teacher_features.shape
    
    # 使用可学习的查询向量
    if not hasattr(self, 'attention_query'):
        self.attention_query = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    query = self.attention_query.expand(batch_size, -1, -1)  # [batch_size, 1, embed_dim]
    key = self.attention_key(teacher_features) if hasattr(self, 'attention_key') else teacher_features
    value = self.attention_value(teacher_features) if hasattr(self, 'attention_value') else teacher_features
    
    # 计算注意力分数
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (embed_dim ** 0.5)
    attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, 1, num_patches]
    
    # 返回注意力权重作为热力图
    return attention_weights.squeeze(1)  # [batch_size, num_patches]