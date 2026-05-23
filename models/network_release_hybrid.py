import torch
from torch import nn
from torch.nn import functional as F

from models.network_release_bak import SPSRNet_release as SPSRNetReleaseBaseline
from models.network_release_refine import SPSRNet_release_refine
from utils import get_gradient


class ConservativeAdaptiveFusionHead(nn.Module):
    # 初始化保守融合头，hidden_dim 表示中间特征通道数，out_channel 表示输出图像通道数。
    def __init__(self, hidden_dim=32, out_channel=1):
        super().__init__()
        # 融合基线分支、改良分支、低分辨率输入、参考输入和结构差异证据。
        self.evidence_fuse = nn.Sequential(
            nn.Conv2d(10, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
        )
        # 预测改良分支应该参与多少，额外输入包含分支分歧和改良收益先验。
        self.blend_gate = nn.Sequential(
            nn.Conv2d(hidden_dim + 3, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True),
        )
        # 在两条分支差异附近再补一个小残差，只做细修正，不推翻基线结果。
        self.delta_refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channel, 3, 1, 1, bias=True),
        )
        # 把初始门控偏向基线分支，避免训练初期就伤到 GUYS 这类本来表现稳定的样本。
        nn.init.zeros_(self.blend_gate[-1].weight)
        nn.init.constant_(self.blend_gate[-1].bias, -2.0)
        # 让额外细修正从 0 开始学习，先靠两条已有分支的互补关系工作。
        nn.init.zeros_(self.delta_refine[-1].weight)
        nn.init.zeros_(self.delta_refine[-1].bias)

    # 前向阶段接收两条分支输出和原始输入，生成保守融合后的超分结果与梯度结果。
    def forward(self, base_sr, refine_sr, base_grad, refine_grad, lr, ref):
        # 把低分辨率输入上采样到输出大小，作为内容保真先验。
        lr_up = F.interpolate(lr, size=base_sr.shape[-2:], mode='bilinear', align_corners=False)
        # 把参考模态上采样到输出大小，作为纹理借用的对照证据。
        ref_up = F.interpolate(ref, size=base_sr.shape[-2:], mode='bilinear', align_corners=False)
        # 计算参考模态梯度，约束融合时不要引入与结构明显冲突的修正。
        ref_grad = get_gradient(ref_up)
        # 计算两条分支在图像域上的差异，分歧越大越需要谨慎融合。
        sr_gap = torch.abs(refine_sr - base_sr)
        # 计算两条分支在梯度域上的差异，用来反映结构是否一致。
        grad_gap = torch.abs(refine_grad - base_grad)
        # 计算基线分支与参考模态的强度差异。
        base_ref_gap = torch.abs(base_sr - ref_up)
        # 计算改良分支与参考模态的强度差异。
        refine_ref_gap = torch.abs(refine_sr - ref_up)
        # 计算基线分支与参考模态在梯度域上的差异。
        base_ref_grad_gap = torch.abs(base_grad - ref_grad)
        # 计算改良分支与参考模态在梯度域上的差异。
        refine_ref_grad_gap = torch.abs(refine_grad - ref_grad)
        # 汇总门控网络需要的主体证据，让网络自己判断何时应该保留 CJH-2、何时借用 CJH-3。
        evidence = torch.cat(
            [
                base_sr,
                refine_sr,
                lr_up,
                ref_up,
                base_grad,
                refine_grad,
                base_ref_gap,
                refine_ref_gap,
                base_ref_grad_gap,
                refine_ref_grad_gap,
            ],
            dim=1,
        )
        # 提取融合门控与细修正共享的中间表示。
        fused_evidence = self.evidence_fuse(evidence)
        # 当改良分支相对基线在参考一致性上更优时，收益先验会更大。
        improvement_prior = torch.relu(
            (base_ref_gap + base_ref_grad_gap) - (refine_ref_gap + refine_ref_grad_gap)
        )
        # 用分支图像分歧、结构分歧和收益先验共同决定改良分支的参与比例。
        blend_gate = torch.sigmoid(
            self.blend_gate(torch.cat([fused_evidence, sr_gap, grad_gap, improvement_prior], dim=1))
        )
        # 额外细修正只允许在两分支已出现差异的位置放大，避免无端扰动基线输出。
        refined_delta = torch.tanh(self.delta_refine(fused_evidence)) * sr_gap
        # 以基线结果为锚点，只在门控允许时注入改良分支的优势和少量细修正。
        sr_out = base_sr + blend_gate * ((refine_sr - base_sr) + refined_delta)
        # 梯度辅助分支也采用同样的保守融合策略，保持训练目标一致。
        grad_out = base_grad + blend_gate * (refine_grad - base_grad)
        # 返回融合后的主输出、辅助梯度输出和门控图，便于后续扩展调试。
        return sr_out, grad_out, blend_gate


class SPSRNet_release_hybrid(nn.Module):
    # 初始化混合网络，保留 CJH-2 基线主干，再叠加 CJH-3 改良分支和自适应融合头。
    def __init__(self, img_size, in_channel=1, out_channel=1, hidden_dim=32, layer_num=3, scale=2, window_size=5,
                 norm_layer=None, act_type='gelu', mode='CNA', upsample=None):
        super().__init__()
        # 基线分支负责维持 GUYS 上已经验证过的稳定高 PSNR。
        self.base_branch = SPSRNetReleaseBaseline(
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
        # 改良分支负责保留 CJH-3 对 HH 困难样本的补偿能力。
        self.refine_branch = SPSRNet_release_refine(
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
        # 融合头使用保守门控把两条分支的优点拼起来，尽量减少此消彼长。
        self.fusion_head = ConservativeAdaptiveFusionHead(
            hidden_dim=max(hidden_dim // 2, 32),
            out_channel=out_channel,
        )

    # 兼容旧 checkpoint 的部分加载，把同一套已有权重同时映射到基线分支和改良分支。
    def load_pretrained(self, state_dict):
        # 读取当前模型参数字典，便于逐项检查键名和尺寸。
        model_state = self.state_dict()
        # 收集所有可以安全加载的权重。
        matched_state = {}
        # 遍历外部 checkpoint 里的每个参数。
        for key, value in state_dict.items():
            # 先尝试直接匹配完整键名。
            if key in model_state and model_state[key].shape == value.shape:
                matched_state[key] = value
            # 再尝试把旧主干权重映射到基线分支。
            base_key = f'base_branch.{key}'
            if base_key in model_state and model_state[base_key].shape == value.shape:
                matched_state[base_key] = value
            # 再尝试把旧主干权重映射到改良分支外层同名参数。
            refine_key = f'refine_branch.{key}'
            if refine_key in model_state and model_state[refine_key].shape == value.shape:
                matched_state[refine_key] = value
            # 最后把旧主干权重映射到改良分支内部 backbone，保证两条分支共享同一初始化起点。
            refine_backbone_key = f'refine_branch.backbone.{key}'
            if refine_backbone_key in model_state and model_state[refine_backbone_key].shape == value.shape:
                matched_state[refine_backbone_key] = value
        # 按非严格模式加载可匹配部分，保留新的融合头随机初始化。
        self.load_state_dict(matched_state, strict=False)
        # 返回命中统计，便于训练和测试入口打印加载信息。
        return len(matched_state), len(model_state), len(state_dict)

    # 前向阶段先分别跑两条分支，再做保守融合输出最终结果。
    def forward(self, x):
        # 拆出低分辨率输入和参考模态输入。
        lr, ref = x
        # 先跑 CJH-2 基线分支，得到稳态主输出和梯度输出。
        base_sr, base_grad = self.base_branch([lr, ref])
        # 再跑 CJH-3 改良分支，得到对困难样本更友好的候选结果。
        refine_sr, refine_grad = self.refine_branch([lr, ref])
        # 用输入自适应门控把两条分支保守融合，尽量同时兼顾 GUYS 和 HH。
        sr_out, grad_out, _ = self.fusion_head(base_sr, refine_sr, base_grad, refine_grad, lr, ref)
        # 返回与原训练接口一致的主图像输出和梯度辅助输出。
        return [sr_out, grad_out]
