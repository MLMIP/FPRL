import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoTemporalProjector(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=4096, output_dim=128, momentum=0.996):
        super().__init__()
        self.momentum = momentum

        # ========== 主干与教师结构 ==========
        self.student_pool = self._build_pool(input_dim, hidden_dim)
        self.teacher_pool = self._build_pool(input_dim, hidden_dim)
        self.student_projector = self._build_projector(hidden_dim, output_dim)
        self.teacher_projector = self._build_projector(hidden_dim, output_dim)

        # 初始化教师参数 = 学生参数
        for t, s in zip(self.teacher_pool.parameters(), self.student_pool.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False
        for t, s in zip(self.teacher_projector.parameters(), self.student_projector.parameters()):
            t.data.copy_(s.data)
            t.requires_grad = False

    def _build_pool(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

    def _build_projector(self, hidden_dim, output_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    @torch.no_grad()
    def _update_teacher(self):
        # 动量更新教师参数
        for tp, sp in zip(self.teacher_pool.parameters(), self.student_pool.parameters()):
            tp.data = tp.data * self.momentum + sp.data * (1 - self.momentum)
        for tp, sp in zip(self.teacher_projector.parameters(), self.student_projector.parameters()):
            tp.data = tp.data * self.momentum + sp.data * (1 - self.momentum)

    def weighted_pool(self, features, attn_weights=None):
        """基于注意力的全局特征聚合"""
        if attn_weights is not None:
            attn_weights = attn_weights.mean(dim=1)  # [B, N]
            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
            pooled = torch.einsum('bn,bnc->bc', attn_weights, features)
        else:
            pooled = features.mean(dim=1)
        return pooled

    def forward(self, past_fs, current_fs, future_fs,
                past_attn=None, current_attn=None, future_attn=None):
        """
        输入:
            *_fs: [B, N, C]
            *_attn: [B, N, N]
        输出:
            h_student: 当前帧学生特征
            h_teacher_past / h_teacher_future: 过去和未来的teacher特征（stop-grad）
        """
        # ---- 更新教师参数 ----
        with torch.no_grad():
            self._update_teacher()

        # ---- 学生分支 ----
        z_current_s = self.weighted_pool(current_fs, current_attn)
        z_current_s = self.student_pool(z_current_s)
        h_current = self.student_projector(z_current_s)  # [B, D]

        # ---- 教师分支（不参与反传）----
        with torch.no_grad():
            z_past_t = self.weighted_pool(past_fs, past_attn)
            z_past_t = self.teacher_pool(z_past_t)
            h_past_t = self.teacher_projector(z_past_t)

            z_future_t = self.weighted_pool(future_fs, future_attn)
            z_future_t = self.teacher_pool(z_future_t)
            h_future_t = self.teacher_projector(z_future_t)

        return h_current, h_past_t.detach(), h_future_t.detach()


class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_past_t, h_current, h_future_t):
        """student-current 对 teacher-past/future 的对比"""
        h_current = F.normalize(h_current, dim=-1)
        h_past_t = F.normalize(h_past_t, dim=-1)
        h_future_t = F.normalize(h_future_t, dim=-1)

        loss_pc = self.info_nce(h_current, h_past_t)
        loss_cf = self.info_nce(h_current, h_future_t)

        return (loss_pc + loss_cf) / 2

    def info_nce(self, anchor, positive):
        # anchor: 学生当前帧
        # positive: 教师过去或未来帧
        batch_size = anchor.size(0)
        sim_matrix = torch.exp(anchor @ positive.t() / self.temperature)  # [B, B]
        pos = torch.diag(sim_matrix)
        neg = sim_matrix.sum(dim=1) - pos
        loss = -torch.log(pos / (pos + neg + 1e-8))
        return loss.mean()
