import math

import torch
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn import functional as F

from utils import PixelShuffleBlock, UpConvBlock, conv_block, get_gradient, Select
from einops import rearrange


class ResidualDenseBlock5C(nn.Module):
    def __init__(self, in_channel, kernel_size=3, hidden_dim=32, stride=1, bias=True, pad_type='zero', norm_type=None,
                 act_type='gelu', mode='CNA'):
        super(ResidualDenseBlock5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = conv_block(in_channel, hidden_dim, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv2 = conv_block(in_channel + hidden_dim, hidden_dim, kernel_size, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv3 = conv_block(in_channel + 2 * hidden_dim, hidden_dim, kernel_size, stride, bias=bias,
                                pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv4 = conv_block(in_channel + 3 * hidden_dim, hidden_dim, kernel_size, stride, bias=bias,
                                pad_type=pad_type,
                                norm_type=norm_type, act_type=act_type, mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(in_channel + 4 * hidden_dim, in_channel, 3, stride, bias=bias, pad_type=pad_type,
                                norm_type=norm_type, act_type=last_act, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x


class RRDBx(nn.Module):
    def __init__(self, in_channel, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                 norm_type=None, act_type='gelu', mode='CNA'):
        super(RRDBx, self).__init__()
        self.stack_num = stack_num
        self.RRDBx = nn.ModuleList([
            ResidualDenseBlock5C(in_channel, kernel_size, hidden_dim, stride, bias, pad_type, norm_type, act_type, mode) for _
            in range(self.stack_num)])

    def forward(self, x):
        for RRDB in self.RRDBx:
            x = RRDB(x)
        return x.mul(0.2) + x


class FeatureAdaption(nn.Module):
    def __init__(self, in_channel=32, use_residual=True, learnable=True):
        super(FeatureAdaption, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(in_channel, affine=False)
        if self.learnable:
            self.conv_1 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 3, 1, 1, bias=True),
                                        nn.GELU())
            self.conv_gamma = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True)
            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref_ori):  # lr (b,32,120,120)  ref (b,32,240,240)
        b, c, h, w = lr.shape
        lr_mean = torch.mean(lr.reshape(b, c, h * w), dim=-1, keepdim=True).reshape(b, c, 1, 1)
        lr_std = torch.std(lr.reshape(b, c, h * w), dim=-1, keepdim=True).reshape(b, c, 1, 1)
        ref_normed = self.norm_layer(ref_ori)  # b,32,120,120
        style = self.conv_1(torch.cat([lr, ref_ori], dim=1))  # b,32,120,120
        gamma = self.conv_gamma(style)
        beta = self.conv_beta(style)
        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
        out = ref_normed * gamma + beta
        return out


class DifferenceAwareRegistration(nn.Module):
    def __init__(self, in_channel=32, hidden_dim=32, max_offset=0.20):
        super(DifferenceAwareRegistration, self).__init__()
        # 记录归一化形变范围，避免跨模态对齐时出现过大的错位采样。
        self.max_offset = max_offset
        # 用目标结构、参考结构和粗差异共同预测柔性偏移场。
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channel * 3, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 3, 1, 1, bias=True),
        )
        # 用对齐后的残差生成结构差异图，作为后续可信迁移的抑制信号。
        self.diff_head = nn.Sequential(
            nn.Conv2d(in_channel * 3, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_grid(x):
        # 读取特征图尺寸，为网格采样准备归一化坐标。
        _, _, h, w = x.shape
        # 生成纵向坐标。
        grid_y = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        # 生成横向坐标。
        grid_x = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        # 组合成二维坐标网格。
        mesh_y, mesh_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
        # 拼出 grid_sample 需要的 x-y 排列。
        base_grid = torch.stack([mesh_x, mesh_y], dim=-1)
        # 扩展到 batch 维度。
        return base_grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

    def forward(self, target, ref):
        # 先计算未对齐时的绝对差异，给偏移估计提供结构提示。
        coarse_diff = torch.abs(target - ref)
        # 拼接目标、参考和差异特征来预测偏移场。
        offset = self.offset_head(torch.cat([target, ref, coarse_diff], dim=1))
        # 用 tanh 限制偏移范围，提升错位场景下的稳定性。
        offset = torch.tanh(offset) * self.max_offset
        # 构造基础采样网格。
        base_grid = self._build_grid(target)
        # 将偏移场转换为 grid_sample 的最后一维坐标格式。
        flow_grid = base_grid + offset.permute(0, 2, 3, 1)
        # 对参考特征做柔性重采样，实现鲁棒错位对齐。grid_sample就是根据网格，去对应位置上采样的
        aligned_ref = F.grid_sample(ref, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)
        # 重新计算对齐后的结构残差。
        refined_diff = torch.abs(target - aligned_ref)
        # 生成 0 到 1 的差异图，值越大代表当前位置越不可信。
        diff_map = self.diff_head(torch.cat([target, aligned_ref, refined_diff], dim=1))
        # 返回对齐参考、差异图和偏移场，便于后续高频调制使用。
        return aligned_ref, diff_map, offset


class CredibleHighFrequencyModulation(nn.Module):
    def __init__(self, in_channel=32, hidden_dim=32):
        super(CredibleHighFrequencyModulation, self).__init__()
        # 用轻量平滑核近似低频分量，再通过残差显式提取高频。
        self.low_pass = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # 用目标高频、参考高频、差异和高频残差共同估计置信度。
        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channel * 3 + 1, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 对可信高频做一次投影，便于后续和主干特征融合。
        self.refine = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
        )

    def forward(self, target, aligned_ref, diff_map):
        # 提取目标特征的高频残差。
        target_high = target - self.low_pass(target)
        # 提取对齐参考特征的高频残差。
        ref_high = aligned_ref - self.low_pass(aligned_ref)
        # 计算两者的高频差异，抑制不一致的纹理借用。
        high_gap = torch.abs(target_high - ref_high)
        # 拼接多种高频证据来预测可信度图。
        confidence = self.confidence_head(torch.cat([target_high, ref_high, high_gap, diff_map], dim=1))
        # 用差异图进一步压制高风险区域的参考迁移。
        confidence = confidence * (1.0 - diff_map)
        # 只保留高置信的参考高频响应。
        credible_ref_high = ref_high * confidence
        # 将目标高频和可信参考高频做可学习融合。
        fused_high = self.refine(torch.cat([target_high, credible_ref_high], dim=1))
        # 返回融合高频、置信度图和可信参考高频。
        return fused_high, confidence, credible_ref_high


class DualDomainSparseFusion(nn.Module):
    def __init__(self, in_channel=32, hidden_dim=32):
        super(DualDomainSparseFusion, self).__init__()
        # 用空间分支聚合目标特征和可信高频，保留 HFST 的局部结构建模优势。
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channel * 2, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channel, 3, 1, 1, bias=True),
        )
        # 用显式门控生成空间域稀疏掩码，减少低价值连接。
        self.spatial_mask = nn.Sequential(
            nn.Conv2d(in_channel * 2 + 2, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 用频率门控控制参考高频在频域中的注入强度。
        self.frequency_mask = nn.Sequential(
            nn.Conv2d(in_channel * 2 + 2, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )
        # 用动态门控在空间分支和频率分支之间自适应分配权重。
        self.branch_gate = nn.Sequential(
            nn.Linear(in_channel * 2, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, 2, bias=False),
        )
        # 最后一层卷积用于整理双域融合后的响应。
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
        )
        # 记录通道平均池化操作，用于动态权重预测。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, target, fused_high, confidence, diff_map):
        # 拼接目标特征、可信高频、置信度和差异图，构造双域共享证据。
        fusion_evidence = torch.cat([target, fused_high, confidence, diff_map], dim=1)
        # 预测空间域稀疏掩码，只保留更有价值的位置响应。
        spatial_mask = self.spatial_mask(fusion_evidence)
        # 在空间域内融合目标与可信高频。
        spatial_feature = self.spatial_proj(torch.cat([target, fused_high], dim=1))
        # 用稀疏掩码强调关键区域，模拟 CAFET 的稀疏选择思想。
        spatial_feature = spatial_feature * spatial_mask
        # 预测频率域门控，控制参考高频的频域注入强度。
        frequency_mask = self.frequency_mask(fusion_evidence)
        # 取出特征图尺寸，给 rfft/irfft 做形状约束。
        h, w = target.shape[-2:]
        # 将空间门控压缩到频域宽度，作为频域稀疏掩码。
        frequency_mask = F.adaptive_avg_pool2d(frequency_mask * confidence, (h, w // 2 + 1))
        # 对目标特征做二维快速傅里叶变换。
        target_fft = torch.fft.rfft2(target, norm='ortho')
        # 对可信高频做二维快速傅里叶变换。
        high_fft = torch.fft.rfft2(fused_high, norm='ortho')
        # 在频率域内按掩码注入可信参考高频。
        fused_fft = target_fft + high_fft * frequency_mask
        # 将融合后的频域表示变回空间域特征。
        frequency_feature = torch.fft.irfft2(fused_fft, s=(h, w), norm='ortho')
        # 汇聚空间分支描述子。
        spatial_descriptor = self.avg_pool(spatial_feature).flatten(1)
        # 汇聚频率分支描述子。
        frequency_descriptor = self.avg_pool(frequency_feature).flatten(1)
        # 预测双分支的动态组合系数。
        gate = F.softmax(self.branch_gate(torch.cat([spatial_descriptor, frequency_descriptor], dim=1)), dim=1)
        # 对空间分支加权。
        weighted_spatial = spatial_feature * gate[:, 0].view(-1, 1, 1, 1)
        # 对频率分支加权。
        weighted_frequency = frequency_feature * gate[:, 1].view(-1, 1, 1, 1)
        # 融合两个分支，并保留目标残差以维持训练稳定性。
        out = self.out_proj(weighted_spatial + weighted_frequency) + target
        # 返回双域融合后的主干响应。
        return out


class DualDomainConflictAttention(nn.Module):
    def __init__(self, in_channel=32, hidden_dim=32):
        super(DualDomainConflictAttention, self).__init__()
        # 结构自保持门控用目标特征和目标梯度判断模态内证据该保留多少。
        self.structure_gate = nn.Sequential(
            # in_channel * 2 表示目标内容特征和目标梯度特征拼接后的通道数。
            nn.Conv2d(in_channel * 2, hidden_dim, 3, 1, 1, bias=True),
            # GELU 用来增强结构自保持估计的非线性表达。
            nn.GELU(),
            # 输出 1 个空间权重图，表示每个位置的目标自证据强度。
            nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True),
            # Sigmoid 把权重限制在 0 到 1。
            nn.Sigmoid(),
        )
        # 跨模态可靠性门控用内容、梯度、高频置信度和差异图判断参考信息是否可信。
        self.reliability_gate = nn.Sequential(
            # in_channel * 5 + 2 对应目标、参考、双梯度、梯度差异、diff_map 和 confidence。
            nn.Conv2d(in_channel * 5 + 2, hidden_dim, 3, 1, 1, bias=True),
            # GELU 用来学习跨模态可靠性的非线性关系。
            nn.GELU(),
            # 将隐藏特征继续压缩，减少门控预测的参数量。
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1, bias=True),
            # GELU 用来保持门控预测的表达能力。
            nn.GELU(),
            # 输出 1 个空间可靠性图，表示参考模态在当前位置能不能借用。
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1, bias=True),
            # Sigmoid 把可靠性限制在 0 到 1。
            nn.Sigmoid(),
        )
        # 梯度域裁决门控在模态内和模态间注意力冲突时分配优先级。
        self.gradient_conflict_gate = nn.Sequential(
            # in_channel * 3 + 2 对应模态内、模态间、二者差异、结构权重和可靠性权重。
            nn.Conv2d(in_channel * 3 + 2, hidden_dim, 3, 1, 1, bias=True),
            # GELU 用来拟合冲突强度和裁决结果之间的非线性关系。
            nn.GELU(),
            # 输出 2 个 logits，分别对应模态内分支和模态间分支。
            nn.Conv2d(hidden_dim, 2, 1, 1, 0, bias=True),
        )
        # 梯度域细化层把裁决后的模态内/模态间响应整理成 Gradient-IM2CE 输出。
        self.gradient_refine = nn.Sequential(
            # in_channel * 2 对应裁决后的模态内响应和模态间响应。
            nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0, bias=True),
            # GELU 用来增强梯度域融合结果的表达能力。
            nn.GELU(),
            # 3x3 卷积补充局部邻域结构上下文。
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
        )
        # 频率域门控控制可信高频在频谱中的注入强度。
        self.frequency_gate = nn.Sequential(
            # in_channel * 2 + 2 对应目标特征、可信高频、confidence 和 diff_map。
            nn.Conv2d(in_channel * 2 + 2, hidden_dim, 3, 1, 1, bias=True),
            # GELU 用来提升频率域门控的非线性表达。
            nn.GELU(),
            # 输出 1 个空间权重图，后续会压缩到 rfft 的频谱尺寸。
            nn.Conv2d(hidden_dim, 1, 1, 1, 0, bias=True),
            # Sigmoid 把频率注入强度限制在 0 到 1。
            nn.Sigmoid(),
        )
        # 频率域细化层把频域回流结果整理成 Frequency-IM2CE 输出。
        self.frequency_refine = nn.Sequential(
            # in_channel * 2 对应目标特征和频域回流特征。
            nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0, bias=True),
            # GELU 用来增强频率域结果的纹理表达。
            nn.GELU(),
            # 3x3 卷积补充空间邻域细节。
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
        )
        # 双域空间裁决门控在梯度域和频率域冲突时按位置分配权重。
        self.domain_spatial_gate = nn.Sequential(
            # in_channel * 3 + 2 对应梯度域、频率域、二者差异、diff_map 和 confidence。
            nn.Conv2d(in_channel * 3 + 2, hidden_dim, 3, 1, 1, bias=True),
            # GELU 用来拟合双域冲突的空间非线性。
            nn.GELU(),
            # 输出 2 个空间 logits，分别对应梯度域和频率域。
            nn.Conv2d(hidden_dim, 2, 1, 1, 0, bias=True),
        )
        # 双域通道裁决门控补充全局通道级的重要性判断。
        self.domain_channel_gate = nn.Sequential(
            # in_channel * 2 表示梯度域和频率域全局描述子的拼接长度。
            nn.Linear(in_channel * 2, hidden_dim, bias=False),
            # GELU 用来增强通道权重预测能力。
            nn.GELU(),
            # 输出 2 个通道级 logits，分别对应梯度域和频率域。
            nn.Linear(hidden_dim, 2, bias=False),
        )
        # 输出投影层生成最终增强特征。
        self.out_proj = nn.Sequential(
            # 3x3 卷积先细化双域裁决后的融合响应。
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
            # GELU 用来增强最终融合结果的表达。
            nn.GELU(),
            # 3x3 卷积输出与输入同通道的增强特征。
            nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=True),
        )
        # 平均池化用来生成通道裁决所需的全局描述子。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, intra_feature, inter_feature, target_feature, ref_feature, target_grad, ref_grad, fused_high,
                confidence, diff_map):
        # structure_weight 表示目标自身结构在每个空间位置的自保持强度。
        structure_weight = self.structure_gate(torch.cat([target_feature, target_grad], dim=1))
        # grad_gap 表示目标梯度和参考梯度之间的结构差异。
        grad_gap = torch.abs(target_grad - ref_grad)
        # reliability_input 汇总跨模态可靠性估计所需的内容域、梯度域和差异证据。
        reliability_input = torch.cat([target_feature, ref_feature, target_grad, ref_grad, grad_gap, diff_map,
                                       confidence], dim=1)
        # reliability_weight 表示参考模态在每个位置是否适合参与模态间注意力。
        reliability_weight = self.reliability_gate(reliability_input) * (1.0 - diff_map)
        # preserved_intra 用结构自保持权重增强模态内注意力结果。
        preserved_intra = intra_feature * (1.0 + structure_weight)
        # selected_inter 用跨模态可靠性权重筛选模态间注意力结果。
        selected_inter = inter_feature * reliability_weight
        # gradient_conflict 表示模态内和模态间证据之间的响应差异。
        gradient_conflict = torch.abs(preserved_intra - selected_inter)
        # gradient_gate_input 汇总梯度域冲突裁决需要的全部证据。
        gradient_gate_input = torch.cat([preserved_intra, selected_inter, gradient_conflict, structure_weight,
                                         reliability_weight], dim=1)
        # gradient_weights 表示模态内和模态间分支在梯度域中的裁决权重。
        gradient_weights = F.softmax(self.gradient_conflict_gate(gradient_gate_input), dim=1)
        # weighted_intra 表示裁决后的模态内响应。
        weighted_intra = preserved_intra * gradient_weights[:, 0:1]
        # weighted_inter 表示裁决后的模态间响应。
        weighted_inter = selected_inter * gradient_weights[:, 1:2]
        # gradient_feature 表示 Gradient-IM2CE 分支输出。
        gradient_feature = self.gradient_refine(torch.cat([weighted_intra, weighted_inter], dim=1))
        # frequency_input 汇总频率域注意力所需的目标特征、高频置信和差异证据。
        frequency_input = torch.cat([target_feature, fused_high, confidence, diff_map], dim=1)
        # frequency_gate 表示可信高频在每个位置的注入强度。
        frequency_gate = self.frequency_gate(frequency_input)
        # h 和 w 表示当前特征图的高和宽，用于傅里叶反变换恢复尺寸。
        h, w = target_feature.shape[-2:]
        # spectral_gate 表示适配 rfft 频谱宽度后的高频注入权重。
        spectral_gate = F.adaptive_avg_pool2d(frequency_gate * confidence, (h, w // 2 + 1))
        # target_fft 表示目标特征的二维实数频谱。
        target_fft = torch.fft.rfft2(target_feature, norm='ortho')
        # high_fft 表示可信高频特征的二维实数频谱。
        high_fft = torch.fft.rfft2(fused_high, norm='ortho')
        # fused_fft 表示注入可信高频后的频域特征。
        fused_fft = target_fft + high_fft * spectral_gate
        # frequency_back 表示频域融合结果回到空间域后的特征。
        frequency_back = torch.fft.irfft2(fused_fft, s=(h, w), norm='ortho')
        # frequency_feature 表示 Frequency-IM2CE 分支输出。
        frequency_feature = self.frequency_refine(torch.cat([target_feature, frequency_back], dim=1))
        # domain_conflict 表示梯度域和频率域输出之间的差异。
        domain_conflict = torch.abs(gradient_feature - frequency_feature)
        # domain_spatial_input 汇总双域空间裁决需要的响应和差异证据。
        domain_spatial_input = torch.cat([gradient_feature, frequency_feature, domain_conflict, diff_map, confidence],
                                         dim=1)
        # domain_spatial_weights 表示每个位置上梯度域和频率域的空间权重。
        domain_spatial_weights = F.softmax(self.domain_spatial_gate(domain_spatial_input), dim=1)
        # gradient_descriptor 表示梯度域分支的全局通道描述子。
        gradient_descriptor = self.avg_pool(gradient_feature).flatten(1)
        # frequency_descriptor 表示频率域分支的全局通道描述子。
        frequency_descriptor = self.avg_pool(frequency_feature).flatten(1)
        # domain_channel_weights 表示梯度域和频率域的全局通道权重。
        domain_channel_weights = F.softmax(self.domain_channel_gate(torch.cat([gradient_descriptor,
                                                                               frequency_descriptor], dim=1)), dim=1)
        # gradient_weight 表示空间权重和通道权重共同决定的梯度域最终权重。
        gradient_weight = domain_spatial_weights[:, 0:1] * domain_channel_weights[:, 0].view(-1, 1, 1, 1)
        # frequency_weight 表示空间权重和通道权重共同决定的频率域最终权重。
        frequency_weight = domain_spatial_weights[:, 1:2] * domain_channel_weights[:, 1].view(-1, 1, 1, 1)
        # normalizer 用于归一化双域权重，避免数值尺度漂移。
        normalizer = gradient_weight + frequency_weight + 1e-6
        # gradient_weight 被归一化到稳定范围。
        gradient_weight = gradient_weight / normalizer
        # frequency_weight 被归一化到稳定范围。
        frequency_weight = frequency_weight / normalizer
        # fused_feature 表示双域冲突裁决后的融合特征。
        fused_feature = gradient_feature * gradient_weight + frequency_feature * frequency_weight
        # out 表示保留目标残差后的最终增强输出。
        out = self.out_proj(fused_feature) + target_feature
        # 返回最终增强输出给 IM2CE 主路径继续融合。
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, C ,H, W)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = rearrange(x, 'b c (h ws1) (w ws2) -> (b h w) (ws1 ws2) c', ws1=window_size, ws2=window_size)
    # windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, C, H, W)
    """
    # B = int(windows.shape[0] / (H * W / window_size / window_size))
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)

    x = rearrange(windows, '(b h w) (w1 w2) c -> b c (h w1) (w w2)', h=H // window_size, w=W // window_size,
                  w1=window_size, w2=window_size)
    return x


def window_partition_downshuffle(x, window_size):
    """

    :param x: B,C,H,W
    :param window_size: window size
    :return: windows: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    h_interval, w_interval = int(H / window_size), int(W / window_size)
    y = []
    for i in range(h_interval):
        for j in range(w_interval):
            y.append(x[:, :, i::h_interval, j::w_interval])  # fold
    # windows = torch.stack(y, 2).contiguous().view(-1, window_size * window_size, C)
    y = torch.stack(y, 2)
    windows = rearrange(y, 'b c l w1 w2 -> (b l) (w1 w2) c', w1=window_size, w2=window_size)
    return windows


def window_reverse_downshuffle(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    scale = int(pow((H * W / window_size / window_size), 0.5))
    # x = windows.view(B, H // window_size * W // window_size, window_size, window_size, -1).permute(0, 4, 1, 2,
    #                                                                                                3).contiguous(
    # ).view(B, -1, window_size, window_size)

    x = rearrange(windows, '(b l) (w1 w2) c -> b (c l) w1 w2', b=B, w1=window_size, w2=window_size)
    pixshuffle = nn.PixelShuffle(scale)
    x = pixshuffle(x)
    return x


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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LAM_Module(nn.Module):
    """ Layer attention module in HAN (ECCV2020) """
    def __init__(self):
        super(LAM_Module, self).__init__()
        # 论文提到的残差缩放系数，初始为0，训练中学习
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x.shape: [B, M, N, D]
        # 对应论文：B=batch, M=头数(分组数), N=像素数, D=每个头的维度(C/M)
        B, M, N, D = x.size()

        # --------------------------
        # 步骤1：维度重排，为公式做准备
        # --------------------------
        # 原 x: [B, M, N, D] → permute → [B, D, M, N]
        # 为什么这么排？因为公式是按“每个像素n，不同头m/l之间的相关性”计算的
        x = x.permute(0, 3, 1, 2)  # [B, D, M, N]

        # 重塑为二维矩阵，方便后续矩阵乘法
        # proj_query: [B, D, M*N]  （论文里的 \hat{v}^{(m)}[n] 被展平了）
        proj_query = x.view(B, D, -1)
        # proj_key: [B, M*N, D] （转置后，和query做矩阵乘法）
        proj_key = x.view(B, D, -1).permute(0, 2, 1)

        # --------------------------
        # 步骤2：计算公式中的 |v^(m)[n] ∘ v^(l)[n]|
        # --------------------------
        # 矩阵乘法：proj_query @ proj_key = [B, D, D]
        # 这里的矩阵乘法，本质上是在计算所有头之间的元素级乘积（Hadamard乘积）的相似度
        # 对应公式的分子 |v^(m)[n] ∘ v^(l)[n]|
        energy = torch.bmm(proj_query, proj_key)

        # （论文没写这一步，是代码里的数值稳定技巧，防止exp溢出）
        energy_new = energy - torch.max(energy, -1, keepdim=True)[0].expand_as(energy)

        # --------------------------
        # 步骤3：公式的softmax归一化，得到注意力矩阵A[n,m,l]
        # --------------------------
        # 对应公式的 exp(|v^(m)∘v^(l)|) / sum(exp(...))
        attention = self.softmax(energy)  # [B, D, D]

        # --------------------------
        # 步骤4：加权求和，得到论文里的 z^(m)[n]
        # --------------------------
        proj_value = x.view(B, D, -1)  # [B, D, M*N]
        # attention @ proj_value → [B, D, M*N]
        # 对应公式的 z^(m)[n] = sum_l A[n,m,l] * v^(l)[n]
        out = torch.bmm(attention, proj_value)

        # 恢复维度：[B, D, M*N] → [B, D, M, N]
        out = out.view(B, D, M, N)

        # --------------------------
        # 步骤5：残差连接，对应论文“add a residual connection”
        # --------------------------
        # gamma是可学习的缩放系数，控制LAM的贡献强度
        out = self.gamma * out + x

        # 恢复原始输入维度：[B, D, M, N] → [B, M, N, D]
        out = out.permute(0, 2, 3, 1)
        return out


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.layer_att_other = LAM_Module()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if type(x) is tuple:
            x1, x2 = x
        else:
            raise NotImplementedError('{} is not tuple'.format(x))
        B_, N, C = x1.shape
        q = self.q(x1).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        #intra-head correlation
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        #inter-head correlation
        x = self.layer_att_other(attn @ v).transpose(1, 2).reshape(B_, N, C) + q.transpose(1, 2).reshape(B_, N, C) + x1
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=5, shift_size=0, partition_type='window_partition',
                 mlp_ratio=4., scale=1, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.scale = scale
        self.shift_size = shift_size
        if partition_type == 'window_partition':
            self.window_partition = window_partition
            self.window_reverse = window_reverse
        else:
            self.window_partition = window_partition_downshuffle
            self.window_reverse = window_reverse_downshuffle

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.LCLG_1 = nn.Sequential(
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.GELU()
        )
        self.LCLG_2 = nn.Sequential(
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm([input_resolution[0], input_resolution[0]]),
            nn.GELU()
        )
        self.x1_pos_embedding = nn.Parameter(
            torch.ones(1, window_size * window_size, dim))
        self.x2_pos_embedding = nn.Parameter(
            torch.ones(1, window_size * window_size, dim))

        mlp_hidden_dim = int(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        img_mask = img_mask.permute(0, 3, 1, 2)
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size*window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x: list or tuple):
        if type(x) is not list and type(x) is not tuple:
            x1 = x2 = x
        else:
            x1, x2 = x
        B, C, H, W = x1.shape
        x_size = (H, W)
        x1 = self.LCLG_1(x1)
        x2 = self.LCLG_2(x2)
        # cyclic shift
        if self.shift_size > 0:
            shifted_x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_x2 = torch.roll(x2, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x1 = x1
            shifted_x2 = x2

        # partition windows
        # B,C,h,w => B*num,window_size * window_size,C
        x1_windows = self.window_partition(shifted_x1, self.window_size)
        x2_windows = self.window_partition(shifted_x2, self.window_size)

        x1_windows = x1_windows + self.x1_pos_embedding
        x2_windows = x2_windows + self.x2_pos_embedding
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        # nW*B, window_size*window_size, C
        if self.input_resolution == x_size:
            attn_windows = self.attn((x1_windows, x2_windows), mask=self.attn_mask)
        else:
            attn_windows = self.attn((x1_windows, x2_windows), mask=self.calculate_mask(x_size).to(x1.device))

        # FFN
        x = x1_windows + self.drop_path(attn_windows)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        shifted_x = self.window_reverse(x, self.window_size, H, W)  # B C, H W

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x
        return x


class SLCC(nn.Module):
    """
        short long cross conv attention
    """

    def __init__(self, in_channel, img_size, hidden_dim, scale=2, window_size=5):
        super(SLCC, self).__init__()
        self.in_channel = in_channel
        self.scale = scale

        self.LG = RRDBx(in_channel=in_channel * 2, stack_num=1, kernel_size=3, hidden_dim=16)
        self.L_conv1_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.G_conv1_2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)

        self.SWA = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=1,
                              window_size=window_size, shift_size=0, partition_type='window_partition')
        self.LWA = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=1,
                              window_size=window_size, shift_size=0,
                              partition_type='window_partition_downshuffle')
        self.IMA_1 = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=scale,
                                window_size=window_size, shift_size=0)
        self.IMA_2 = BasicBlock(in_channel, input_resolution=(img_size, img_size), num_heads=4, scale=scale,
                                window_size=window_size, shift_size=window_size // 2)
        self.FA = FeatureAdaption(in_channel)
        self.EFF = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, 1, 1, 0),
                                 nn.GELU(),
                                 nn.Conv2d(in_channel, in_channel, 3, 1, 1))
        self.LNA = nn.Sequential(
            nn.InstanceNorm2d(in_channel),
            nn.GELU()
        )
        self.GNA = nn.Sequential(
            nn.InstanceNorm2d(in_channel),
            nn.GELU()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.ADM = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 2, bias=False),
        )
        # 双域冲突注意力把 Gradient-IM2CE 和 Frequency-IM2CE 融入原 IM2CE 内部。
        self.dual_domain_attention = DualDomainConflictAttention(in_channel=in_channel, hidden_dim=hidden_dim)
        self.conv_last = conv_block(in_channel, in_channel, kernel_size=1, stride=1)

    def forward(self, x):
        if type(x) is list:
            x, Fc_in, target_grad, ref_grad, fused_high, confidence, diff_map = x

        fea = torch.cat([x, Fc_in], dim=1)
        B, C, H, W = x.shape
        LG = self.LG(fea).reshape(B, C, 2, H, W).permute(2, 0, 1, 3, 4)
        L, G = LG[0], LG[1]

        L_1_1 = self.SWA(L)
        L_out = self.LWA(L_1_1)

        fa = self.FA(G, Fc_in)

        G = self.EFF(torch.concat([fa, G], dim=1)) + G

        G_1_1 = self.IMA_1([G, fa])
        G_out = self.IMA_2([G_1_1, fa])

        L_out = self.LNA(L_out)
        G_out = self.GNA(G_out)
        y = self.avg_pool(x).view(B, C)
        y = self.ADM(y)

        ax = F.softmax(y, dim=1)

        # 先保留原 IM2CE 的通道级模态内/模态间基础权重，作为双域裁决前的先验。
        L_out = L_out * ax[:, 0].view(B, 1, 1, 1)
        # 再保留原 IM2CE 的跨模态通道级基础权重，避免丢掉原结构中的全局调制能力。
        G_out = G_out * ax[:, 1].view(B, 1, 1, 1)
        # 在注意力内部执行梯度域注意力、频率域注意力和双域冲突裁决融合。
        out = self.dual_domain_attention(L_out, G_out, x, Fc_in, target_grad, ref_grad, fused_high, confidence,
                                         diff_map)
        out = self.conv_last(out)
        out = out + x
        return out


class CohfT(nn.Module):
    def __init__(self, in_channel, img_size, hidden_dim, layer_num, scale=2, window_size=5, kernel_size=3,
                 stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA'):
        super(CohfT, self).__init__()
        self.RRDB1_1 = RRDBx(in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride,
                             bias=bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.conv2_1 = conv_block(2 * in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.SLCC_block = nn.ModuleList(
            [SLCC(in_channel, img_size, hidden_dim, scale=scale, window_size=window_size) for _ in range(layer_num)])

        self.Select = Select(in_nc=in_channel, out_nc=hidden_dim)
        # 错位鲁棒对齐模块先生成对齐参考和结构差异图。
        self.darm = DifferenceAwareRegistration(in_channel=in_channel, hidden_dim=hidden_dim)
        # 可信高频调制模块为 Frequency-IM2CE 提供高频证据和置信度图。
        self.chpm = CredibleHighFrequencyModulation(in_channel=in_channel, hidden_dim=hidden_dim)
        # 新增错位鲁棒对齐模块，借鉴 MAR-DUN 的柔性对齐思想。
        self.darm = DifferenceAwareRegistration(in_channel=in_channel, hidden_dim=hidden_dim)
        # 新增差异感知可信高频调制模块，借鉴 HFMT 与差异学习思路。
        self.chpm = CredibleHighFrequencyModulation(in_channel=in_channel, hidden_dim=hidden_dim)
        # 新增空间-频率双域稀疏融合模块，借鉴 DMASR 与 CAFET 思路。

        # 对齐模块生成参考特征和结构差异图，供嵌入式双域注意力使用。
        self.darm = DifferenceAwareRegistration(in_channel=in_channel, hidden_dim=hidden_dim)
        # 高频模块生成可信高频和置信度图，供 Frequency-IM2CE 使用。
        self.chpm = CredibleHighFrequencyModulation(in_channel=in_channel, hidden_dim=hidden_dim)

    def forward(self, F_in, p_in, Fc_in, factor):
        """
   特征融合与上采样模块的前向传播.

   Args:
       F_in: 主干输入特征，来自当前网络主分支，表示当前阶段待增强的内容特征.
       p_in: 结构先验特征，一般来自梯度/结构分支，用于帮助恢复边缘和结构细节.
       Fc_in: 参考特征，来自参考图像或参考分支，用于提供可借鉴的纹理和高频细节.
       factor: 上采样倍率，当前模块输出后要放大多少倍，通常为2或3.

   Returns:
       融合并上采样后的特征图.
   """
        E = self.RRDB1_1(F_in)
        Fs = self.conv1_1(E)
        B, C, H, W = p_in.shape
        Fs1 = torch.cat([p_in, Fs], dim=1)
        Fs2 = self.conv2_1(Fs1)
        # 先用结构先验支路和参考支路做柔性错位对齐。
        aligned_ref, diff_map, _ = self.darm(p_in, Fc_in)
        # 再从当前主干特征和对齐参考中提取可信高频信息。
        fused_high, confidence, credible_ref_high = self.chpm(Fs2, aligned_ref, diff_map)
        # 将对齐参考和可信高频拼成增强参考上下文，供 HFST 原有块继续建模。
        enhanced_ref = aligned_ref + credible_ref_high
        res_Fs2 = Fs2
        for SLCC in self.SLCC_block:
            # 将双域证据传入每层 IM2CE，使梯度域与频率域注意力在核心注意力内部完成。
            Fs2 = SLCC([Fs2, enhanced_ref, p_in, aligned_ref, fused_high, confidence, diff_map])
        FL0 = res_Fs2 + Fs2[:, :C, :, :]
        s = self.Select(Fs, FL0)

        Fs_out = E + s
        # 插值
        Fs_out = F.interpolate(Fs_out, scale_factor=factor)
        FL0 = F.interpolate(FL0, scale_factor=factor)
        return Fs_out, FL0

'''
Output gate
'''
class HRReconstruction(nn.Module):
    def __init__(self, in_channel, out_channel, scale=2, kernel_size=3, stride=1,
                 act_type=None, norm_type=None,
                 mode='CNA'):
        super(HRReconstruction, self).__init__()
        self.scale = scale
        self.RRDB1_1 = RRDBx(in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride, bias=True,
                             pad_type='zero',
                             norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=None, act_type=None)
        self.RRDB1_2 = RRDBx(2 * in_channel, stack_num=1, kernel_size=kernel_size, hidden_dim=16, stride=stride,
                             bias=True, pad_type='zero', norm_type=norm_type, act_type=act_type, mode=mode)
        self.conv1_2 = conv_block(in_channel * 2, in_channel, kernel_size=kernel_size, norm_type=None,
                                  act_type=act_type)
        self.conv1_3 = conv_block(in_channel, out_channel, kernel_size=kernel_size, norm_type=None, act_type=None)
        # 第二条路
        self.conv2_1 = conv_block(in_channel, in_channel, kernel_size=kernel_size, norm_type=norm_type,
                                  act_type=act_type)
        self.conv2_4 = conv_block(in_channel, out_channel, kernel_size=1, norm_type=None, act_type=act_type)

    def forward(self, F_in, lr, P):
        Fc_in_sr = F.interpolate(lr, scale_factor=self.scale)
        F_in = self.RRDB1_1(F_in)
        F_in = self.conv1_1(F_in)

        P_ = self.conv2_1(P)
        P = P_ + P
        R_out = self.conv2_4(P)

        res = torch.cat([P, F_in], dim=1)
        I_out = self.RRDB1_2(res)
        I_out = self.conv1_2(I_out)
        I_out = self.conv1_3(I_out) + Fc_in_sr
        return I_out, R_out


class SPSRNet_release(nn.Module):
    same = False

    def __init__(self, img_size, in_channel=1, out_channel=1, hidden_dim=32, layer_num=3, scale=2, window_size=5,
                 norm_layer=None, act_type='gelu', mode='CNA', upsample=None):
        super(SPSRNet_release, self).__init__()
        self.scale = scale
        self.fea_conv = conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=norm_layer, act_type=act_type)
        self.Rs_grad_fea = nn.Sequential(
            conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=None, act_type=None),
            RRDBx(hidden_dim, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                  norm_type=norm_layer,
                  act_type=act_type, mode=mode)
        )

        self.Rc_grad_fea = nn.Sequential(
            conv_block(in_channel, hidden_dim, kernel_size=3, norm_type=None, act_type=None),
            RRDBx(hidden_dim, stack_num=1, kernel_size=3, hidden_dim=16, stride=1, bias=True, pad_type='zero',
                  norm_type=norm_layer,
                  act_type=act_type, mode=mode),
        )

        if self.scale == 3:
            self.deep_feature_extract = nn.ModuleList([
                CohfT(hidden_dim, img_size, scale=3, hidden_dim=hidden_dim,
                      kernel_size=3, layer_num=layer_num, window_size=window_size,
                      stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA')]
            )
        else:
            num = self.scale // 2
            self.deep_feature_extract = nn.ModuleList([
                CohfT(hidden_dim, img_size * 2 ** i, scale=2, hidden_dim=hidden_dim,
                      kernel_size=3, layer_num=layer_num, window_size=window_size,
                      stride=1, bias=True, pad_type='zero', norm_type=None, act_type=None, mode='CNA')
                for i in range(num)])
        self.HR_Reconstruction = HRReconstruction(in_channel=hidden_dim, out_channel=out_channel, scale=scale,
                                                  act_type=act_type, norm_type=norm_layer, mode=mode)

        self.grad = get_gradient

    def forward(self, x: list):
        if type(x) is list:
            lr, Ref = x
        else:
            raise NotImplementedError('input must be two element and type is list but found {}'.format(type(x)))
        if lr.shape == Ref.shape:
            Ref = F.interpolate(Ref, size=(lr.shape[2] * self.scale, lr.shape[3] * self.scale))
        Rs = self.grad(lr)
        F0 = self.fea_conv(lr)
        P = self.Rs_grad_fea(Rs)
        Ref = self.Rc_grad_fea(Ref)
        F_in = F0.clone()
        B, C, H, W = F_in.shape
        for i, Cohf_T in enumerate(self.deep_feature_extract):
            if self.scale == 3:
                ref = F.interpolate(Ref, size=(H, W))
                factor = 3
            else:
                ref = F.interpolate(Ref, scale_factor=0.5 ** (self.scale // 2 - i))
                factor = 2
            F_in, P = Cohf_T(F_in, P, ref, factor)

        # output block
        I_out, R_out = self.HR_Reconstruction(F_in, lr, P)

        return [I_out, R_out]


if __name__ == "__main__":
    from thop import profile, clever_format

    device = 'cuda:0'
    net = SPSRNet_release(img_size=80, in_channel=1, out_channel=1, hidden_dim=32, scale=3, window_size=5,
                                layer_num=1).to(
        device)
    t2_gra = torch.randn(1, 1, 80, 80).to(device)
    t1_gra = torch.randn(1, 1, 240, 240).to(device)
    a = net([t2_gra, t1_gra])
    print(a[0].shape)
    print(a[1].shape)

    flops, params = profile(net, inputs=([t2_gra, t1_gra],))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)
