import torch
from torch import nn
from torch.nn import functional as F

from models.network_release_bak import SPSRNet_release as SPSRNetReleaseBackbone
from utils import conv_block, get_gradient


class ResidualReliabilityRefineHead(nn.Module):
    # 初始化轻量细化头，hidden_dim 表示中间特征通道数，out_channel 表示输出图像通道数。
    def __init__(self, hidden_dim=32, out_channel=1):
        super().__init__()
        # 融合主重建、上采样低分辨率、参考模态和梯度先验，输入通道数固定为 5。
        self.feature_fuse = nn.Sequential(
            nn.Conv2d(5, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        # 根据跨模态差异和梯度差异生成可靠性门控，抑制错误纹理迁移。
        self.reliability_gate = nn.Sequential(
            nn.Conv2d(7, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 预测最终残差图，只对主图像分支做补偿，不改动梯度辅助分支。
        self.residual_predict = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, bias=True),
        )

    # 前向阶段接收基线重建结果、低分辨率输入、参考图像和基线梯度输出。
    def forward(self, sr_base, lr, ref, grad_base):
        # 把低分辨率输入上采样到超分输出大小，作为内容保真先验。
        lr_up = F.interpolate(lr, size=sr_base.shape[-2:], mode='bilinear', align_corners=False)
        # 把参考模态也对齐到输出大小，作为可借用纹理来源。
        ref_up = F.interpolate(ref, size=sr_base.shape[-2:], mode='bilinear', align_corners=False)
        # 计算参考模态的梯度图，用于和主分支梯度做一致性约束。
        ref_grad = get_gradient(ref_up)
        # 聚合细化残差需要的主体证据。
        refine_feature = self.feature_fuse(torch.cat([sr_base, lr_up, ref_up, grad_base, ref_grad], dim=1))
        # 计算主分支与参考分支的强度差异，显式度量跨模态风险。
        image_gap = torch.abs(sr_base - ref_up)
        # 计算主分支梯度与参考梯度的差异，显式度量边缘是否可信。
        grad_gap = torch.abs(grad_base - ref_grad)
        # 结合主体证据与差异证据预测可靠性门控。
        gate = self.reliability_gate(torch.cat([sr_base, lr_up, ref_up, grad_base, ref_grad, image_gap, grad_gap], dim=1))
        # 只在高可靠区域注入残差补偿，减少错误高频放大。
        residual = self.residual_predict(refine_feature) * gate
        # 返回细化残差和门控图，便于后续调试或扩展。
        return residual, gate


class SPSRNet_release_refine(nn.Module):
    # 初始化“作者基线 + 轻量可靠性细化头”，兼顾可复现性和可提升空间。
    def __init__(self, img_size, in_channel=1, out_channel=1, hidden_dim=32, layer_num=3, scale=2, window_size=5,
                 norm_layer=None, act_type='gelu', mode='CNA', upsample=None):
        super().__init__()
        # 主干直接复用作者原版结构，保证和原版权重的主体参数可对齐。
        self.backbone = SPSRNetReleaseBackbone(
            img_size=img_size,
            in_channel=in_channel,
            out_channel=out_channel,
            hidden_dim=hidden_dim,
            layer_num=layer_num,
            scale=scale,
            window_size=window_size,
            norm_layer=norm_layer,
            act_type=act_type,
            mode=mode,
            upsample=upsample,
        )
        # 细化头只在主图像输出后做轻量补偿，避免像当前实验网络那样一次性引入过多新模块。
        self.refine_head = ResidualReliabilityRefineHead(hidden_dim=max(hidden_dim // 2, 32), out_channel=out_channel)

    # 兼容部分加载作者原版权重，把未带 backbone. 前缀的权重自动映射到主干。
    def load_pretrained(self, state_dict):
        # 读取当前模型参数字典，后续用于做键名映射和尺寸校验。
        model_state = self.state_dict()
        # 收集可安全加载的参数，避免新细化头因尺寸不匹配报错。
        matched_state = {}
        # 遍历外部权重里的所有参数。
        for key, value in state_dict.items():
            # 先尝试直接匹配完整键名。
            if key in model_state and model_state[key].shape == value.shape:
                matched_state[key] = value
                continue
            # 再尝试把作者原版键名映射到 backbone. 前缀下。
            backbone_key = f'backbone.{key}'
            if backbone_key in model_state and model_state[backbone_key].shape == value.shape:
                matched_state[backbone_key] = value
        # 按非严格模式加载可匹配部分，保留新细化头随机初始化。
        self.load_state_dict(matched_state, strict=False)
        # 返回命中数量，便于外部打印加载统计。
        return len(matched_state), len(model_state), len(state_dict)

    # 前向阶段先跑作者基线，再做轻量可靠性细化。
    def forward(self, x):
        # 读取低分辨率主输入和参考模态输入。
        lr, ref = x
        # 先得到作者基线的主重建结果和梯度辅助结果。
        sr_base, grad_base = self.backbone([lr, ref])
        # 预测仅在可靠区域生效的残差补偿。
        residual, _ = self.refine_head(sr_base, lr, ref, grad_base)
        # 把残差叠加回主输出，得到最终超分结果。
        sr_out = sr_base + residual
        # 继续复用作者原有的梯度辅助输出，保持训练目标稳定。
        return [sr_out, grad_base]
