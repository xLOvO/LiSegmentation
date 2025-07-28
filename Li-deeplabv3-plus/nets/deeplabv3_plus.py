import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

# ----------------------------------------------------------#
#   Backbone Networks (特征提取主干网络)
# ----------------------------------------------------------#
class MobileNetV2(nn.Module):
    """改进版MobileNetV2主干网络，适配DeepLabv3+架构
    Args:
        downsample_factor: 网络总步长 (8或a16)
        pretrained: 是否加载ImageNet预训练权重
    """

    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        # 加载原始MobileNetV2并截断分类层
        model = mobilenetv2(pretrained)
        # 保留除最后分类层外的所有特征层（移除最后的1x1卷积和池化）
        self.features = model.features[:-1]

        # 网络结构关键点索引
        self.total_idx = len(self.features)  # 特征层总数
        self.down_idx = [2, 4, 7, 14]  # 原始模型中执行下采样的位置索引

        # ------------------------------#
        #   空洞卷积策略调整
        # 目标：控制最终特征图的下采样率
        # ------------------------------#
        if downsample_factor == 8:
            # 步长8配置：最后两个block保持高分辨率
            # 阶段3（索引7-14）：膨胀系数2替代下采样
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            # 阶段4（索引14+）：膨胀系数4进一步扩大感受野
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            # 步长16配置：仅最后block使用空洞卷积
            # 阶段4（索引14+）：膨胀系数2保持感受野
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        """将下采样卷积转换为空洞卷积以保持特征图尺寸

        工作原理：
        - 对stride=2的卷积：移除下采样，通过空洞保持感受野
        - 对普通卷积：增加空洞率扩大感受野

        Args:
            m: 待修改的nn.Conv2d层
            dilate: 目标膨胀系数
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # 处理下采样卷积（stride=2）
            if m.stride == (2, 2):
                m.stride = (1, 1)  # 取消下采样
                if m.kernel_size == (3, 3):
                    # 计算填充保持空间维度：dilation//2 替代原padding=1
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # 处理标准卷积（stride=1）
            elif m.kernel_size == (3, 3):
                # 直接增加空洞率扩大感受野
                m.dilation = (dilate, dilate)
                m.padding = (dilate, dilate)

    def forward(self, x):
        """前向传播返回多级特征

        设计说明：
        - 浅层特征：前4层输出（高分辨率，空间细节丰富）
        - 深层特征：后续层输出（高语义信息，低分辨率）

        Returns:
            low_level_features: [N, 24, H/4, W/4] 浅层细节特征
            x: [N, 320, H/8, W/8] 深层语义特征
        """
        # 提取前4层作为浅层特征（用于后续特征融合）
        low_level_features = self.features[:4](x)
        # 剩余层提取深层语义特征
        x = self.features[4:](low_level_features)
        return low_level_features, x


# ----------------------------------------------------------#
# Attention Mechanisms(注意力机制模块)
# ----------------------------------------------------------#
class MCBAM(nn.Module):
    """通道注意力增强模块 (Channel Attention Boosting)
    Args:
        in_channels: 输入特征图的通道数
        ratios: 通道压缩比例列表 (默认[4,8,16,32])
    """

    def __init__(self, in_channels, ratios=[4, 8, 16, 32]):
        super().__init__()
        # 统一分支输出通道数（确保至少4通道）
        self.base_channels = max(4, in_channels // min(ratios))

        # ------------------------------#
        #   多分支通道注意力设计
        # 不同压缩比捕捉多尺度通道信息
        # ------------------------------#
        self.branches = nn.ModuleList([
            nn.Sequential(
                # 通道信息聚合
                nn.AdaptiveAvgPool2d(1),  # 全局平均池化
                # 通道压缩
                nn.Conv2d(in_channels, max(4, in_channels // r), 1),
                nn.BatchNorm2d(max(4, in_channels // r)),
                nn.ReLU(),
                # 通道对齐（统一各分支输出维度）
                nn.Conv2d(max(4, in_channels // r), self.base_channels, 1),
                nn.BatchNorm2d(self.base_channels),
                nn.ReLU()
            ) for r in ratios  # 遍历所有压缩比例
        ])

        # ------------------------------#
        #   动态分支选择器
        # 学习各分支的重要性权重
        # ------------------------------#
        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 空间信息聚合
            nn.Conv2d(in_channels, 32, 1),  # 特征变换
            nn.ReLU(),
            nn.Conv2d(32, len(ratios), 1)  # 输出分支权重 [B, K, 1, 1]
        )

        # ------------------------------#
        #   轻量空间注意力模块
        # 设计原则：计算高效
        # ------------------------------#
        # 特征统计层（通道维度压缩）
        self.stat_compress = nn.Conv2d(2, 1, kernel_size=1)  # 均值+最大值→单通道
        # 空间注意力生成（3x3卷积捕捉局部关系）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 扩大感受野
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),  # 空间权重生成
            nn.Sigmoid()  # 归一化为[0,1]
        )

        # 残差增强系数（可学习参数）
        self.beta = nn.Parameter(torch.tensor(0.5))  # 初始值0.5

        # 通道数调整层（对齐输入维度）
        self.channel_adjust = nn.Conv2d(self.base_channels, in_channels, 1)

    def forward(self, x):
        """前向传播过程

        处理流程：
        1. 动态分支加权 → 2. 空间注意力生成 → 3. 特征融合 → 4. 残差增强

        Returns:
            增强后的特征图 (与输入同维度)
        """
        # 阶段1: 动态分支融合
        # 计算分支权重 [B, K, 1, 1]
        weights = F.softmax(self.selector(x), dim=1)
        channel_att = 0
        for i, branch in enumerate(self.branches):
            # 加权融合各分支输出
            channel_att += weights[:, i:i + 1] * branch(x)

        # 阶段2: 空间注意力生成
        # 通道统计特征 (均值+最大值)
        avg_map = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_map, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        # 特征融合与压缩
        stat = torch.cat([avg_map, max_map], dim=1)  # [B,2,H,W]
        compressed_stat = self.stat_compress(stat)  # [B,1,H,W]
        # 生成空间注意力图
        spatial_att = self.spatial_att(compressed_stat)  # [B,1,H,W]

        # 阶段3: 特征融合 (通道注意力 ⊗ 空间注意力)
        att = self.channel_adjust(channel_att * spatial_att)

        # 阶段4: 残差增强 (自适应强度控制)
        # 公式: output = x * (1 + β·att), β ∈ (0,1)
        return x * (1 + self.beta.sigmoid() * att)


# ----------------------------------------------------------#
#   Deformable Convolution (可变形卷积)
# ----------------------------------------------------------#
class DeformConv2d(nn.Module):
    """可变形卷积实现 (Deformable Convolution v1/v2)

    特点：
    v1: 仅学习采样位置偏移
    v2: 添加调制机制 (modulation) - 学习每个采样点的权重

    Args:
        inc: 输入通道数
        outc: 输出通道数
        kernel_size: 卷积核尺寸 (默认3)
        padding: 填充大小 (默认1)
        stride: 步长 (默认1)
        bias: 是否添加偏置
        modulation: 是否启用调制机制 (Deformable v2)
    """

    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # 零填充保持空间尺寸
        self.zero_padding = nn.ZeroPad2d(padding)
        # 主卷积层（注意：实际步长为kernel_size）
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size,
                              stride=kernel_size, bias=bias)

        # 偏移量预测网络 (输出2*kernel_size*kernel_size通道)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size,
                                kernel_size=3, padding=1, stride=stride)
        # 初始化偏移量为零
        nn.init.constant_(self.p_conv.weight, 0)

        # 调制机制开关
        self.modulation = modulation
        if modulation:
            # 调制权重预测网络 (输出kernel_size*kernel_size通道)
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size,
                                    kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)

    def forward(self, x):
        """前向传播过程

        步骤:
        1. 预测采样点偏移量
        2. 生成调制权重 (如果启用v2)
        3. 计算采样网格
        4. 双线性插值获取特征
        5. 应用调制权重 (如果启用)
        6. 执行常规卷积

        图示:
        input → [偏移预测] → offset → [网格生成] → p
                   ↓ (v2)
                [权重预测] → m → sigmoid
        """
        # 步骤1: 预测偏移量 [B, 2*N, H, W]
        offset = self.p_conv(x)
        # 步骤2: 预测调制权重 (v2) [B, N, H, W]
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = ks * ks  # 采样点总数

        # 边界填充 (保持后续索引有效性)
        if self.padding:
            x = self.zero_padding(x)

        # 步骤3: 计算采样点坐标 [B, 2N, H, W]
        p = self._get_p(offset, dtype)

        # 步骤4: 双线性插值采样
        # 重排维度: [B, H, W, 2N]
        p = p.contiguous().permute(0, 2, 3, 1)

        # 计算四个角点坐标 (左上、右下、左下、右上)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        # 边界约束 (确保索引不越界)
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # 约束采样点在有效范围内
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], dim=-1)

        # 计算双线性权重 (g)
        # 公式: g = (1 - Δx)(1 - Δy)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 获取四个角点的特征值 [B, C, H, W, N]
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 加权融合 [B, C, H, W, N]
        x_offset = (g_lt.unsqueeze(dim=1) * x_q_lt +
                    g_rb.unsqueeze(dim=1) * x_q_rb +
                    g_lb.unsqueeze(dim=1) * x_q_lb +
                    g_rt.unsqueeze(dim=1) * x_q_rt)

        # 步骤5: 应用调制权重 (Deformable v2)
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)  # [B, H, W, N]
            m = m.unsqueeze(dim=1)  # [B, 1, H, W, N]
            m = m.expand_as(x_offset)  # [B, C, H, W, N]
            x_offset *= m  # 调制特征

        # 步骤6: 重塑并执行卷积
        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        """生成卷积核相对坐标网格

        示例 (3x3核):
        p_n_x = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        p_n_y = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing='ij'
        )
        # 合并坐标: [2N]
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)  # [1, 2N, 1, 1]
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        """生成基础坐标网格 (无偏移时)"""
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing='ij'
        )
        # 扩展为每个采样点 [1, 2N, H, W]
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype):
        """计算实际采样点坐标

        公式: p = p_0 + p_n + offset
        其中:
          p_0: 基础网格坐标
          p_n: 核内相对坐标
          offset: 学习的偏移量
        """
        N = offset.size(1) // 2  # 每个位置包含(x,y)偏移
        h, w = offset.size(2), offset.size(3)

        # 基础坐标网格 [1, 2N, H, W]
        p_0 = self._get_p_0(h, w, N, dtype)
        # 核内相对坐标 [1, 2N, 1, 1]
        p_n = self._get_p_n(N, dtype)
        # 实际采样点 = 基础位置 + 相对偏移 + 学习偏移
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        """双线性插值的索引获取

        Args:
            q: 采样点坐标 [B, H, W, 2N]
            N: 每个位置的采样点数 (k*k)

        Returns:
            插值特征 [B, C, H, W, N]
        """
        b, h, w, _ = q.size()
        padded_w = x.size(3)  # 填充后的宽度
        c = x.size(1)

        # 将特征图展平 [B, C, H*W]
        x = x.contiguous().view(b, c, -1)

        # 计算一维索引: index = x_coord * width + y_coord
        index = q[..., :N] * padded_w + q[..., N:]
        # 扩展索引到所有通道 [B, C, H*W*N]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1)
        index = index.contiguous().view(b, c, -1)

        # 收集特征值 [B, C, H*W*N]
        x_offset = x.gather(dim=-1, index=index)
        # 重塑为 [B, C, H, W, N]
        x_offset = x_offset.view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        """重组特征图用于标准卷积

        将采样点重组为 (ks*ks) 的网格排列

        示例: 3x3核 → 9个采样点 → 重组为3x3网格
        """
        b, c, h, w, N = x_offset.size()
        # 按核尺寸分组 [B, C, H, W*ks, ks]
        x_offset = [x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks)
                    for s in range(0, N, ks)]
        # 合并组 [B, C, H*ks, W*ks]
        x_offset = torch.cat(x_offset, dim=-1)
        x_offset = x_offset.view(b, c, h * ks, w * ks)
        return x_offset


class ADPConv(nn.Module):
    """深度可分离卷积增强模块 (Depthwise-Pointwise Convolution Enhancement)
    Args:
        c_in: 输入通道数
        c_out: 输出通道数
        k_size: 可变形卷积核大小 (默认3)
    """

    def __init__(self, c_in, c_out, k_size=3):
        super().__init__()
        # ------------------------------#
        #   阶段1: 通道优化 (通道投影)
        # 功能: 跨通道特征重组与维度调整
        # 优势: 1x1卷积高效实现通道间交互
        # ------------------------------#
        self.pw = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1),  # 点卷积 (无空间聚合)
            nn.BatchNorm2d(c_out),  # 归一化加速收敛
            nn.ReLU()  # 引入非线性
        )

        # ------------------------------#
        #   阶段2: 自适应空间采样
        # 功能: 几何变形感知的特征提取
        # 核心: 可变形卷积学习自适应采样位置
        # ------------------------------#
        self.deform = DeformConv2d(c_out, c_out, k_size)  # 空间自适应卷积

        # ------------------------------#
        #   阶段3: 特征校准 (通道注意力)
        # 设计: 瓶颈结构(bottleneck)降低计算量
        # 输出: 空间敏感的特征权重图[0,1]
        # ------------------------------#
        self.calibrate = nn.Sequential(
            nn.Conv2d(c_out, c_out // 4, 1),  # 通道压缩
            nn.ReLU(),  # 非线性激活
            nn.Conv2d(c_out // 4, c_out, 1),  # 通道恢复
            nn.Sigmoid()  # 归一化为注意力权重[0,1]
        )

    def forward(self, x):
        """前向传播流程

        数据处理流:
        输入 → [通道优化] → 特征A → [自适应采样] → 特征B
                         ↘ [特征校准] → 注意力图 → [加权融合] → 输出

        数学表达:
        output = DeformConv(PW(x)) * σ(Calibrate(PW(x)))

        设计说明:
        1. 共享阶段1的输出，减少计算冗余
        2. 可变形卷积增强空间特征提取能力
        3. 校准机制抑制噪声，增强关键特征
        """
        # 阶段1: 通道优化 (维度调整)
        x = self.pw(x)  # [B, c_in, H, W] → [B, c_out, H, W]

        # 阶段2: 自适应空间采样 (几何感知特征提取)
        deform_out = self.deform(x)  # 学习采样位置 [B, c_out, H, W]

        # 阶段3: 特征校准 (生成注意力图)
        attn = self.calibrate(x)  # 空间敏感权重 [B, c_out, H, W] ∈ [0,1]

        # 特征加权: 增强重要特征，抑制噪声
        return deform_out * attn  # [B, c_out, H, W]


# ----------------------------------------------------------#
#   Feature Fusion Modules (特征融合模块)
# ----------------------------------------------------------#
class ContextGating(nn.Module):
    """高效交叉门控机制

    功能描述：
    1. 基于上下文特征动态生成卷积核参数
    2. 使用生成核进行特征门控加权
    3. 全向量化实现避免显式循环

    设计优势：
    - 上下文感知：根据上下文动态调整门控参数
    - 计算高效：unfold+向量化操作替代逐像素循环
    - 空间自适应：为每个位置生成专属卷积核

    Args:
        dim: 特征通道维度
    """

    def __init__(self, dim):
        super().__init__()
        # 动态卷积核生成器
        self.dynamic_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局上下文聚合
            nn.Conv2d(dim, max(8, dim // 32), 1),  # 瓶颈压缩
            nn.ReLU(),  # 引入非线性
            nn.Conv2d(max(8, dim // 32), 9, 1)  # 输出3x3核参数 [B, 9, 1, 1]
        )
        self.sigmoid = nn.Sigmoid()  # 门激活函数

    def forward(self, feat, context):
        """前向传播

        处理流程：
        1. 基于上下文生成动态卷积核
        2. 反射填充特征图
        3. 展开特征图为局部块
        4. 向量化点乘实现动态卷积
        5. 应用sigmoid门控

        数学表达：
        gate = σ(Σ(Unfold(feat) * kernel_params))
        output = feat ⊙ gate

        维度变换：
        feat: [B, C, H, W] → 输出: [B, C, H, W]
        context: [B, C, H, W] → 核参数: [B, 9, 1, 1]
        """
        # 阶段1: 动态卷积核生成
        # 基于上下文特征生成3x3核参数 [B, 9, 1, 1]
        kernel_params = self.dynamic_conv(context)
        b, c, h, w = feat.shape

        # 阶段2: 特征图预处理
        # 反射填充 (保持边缘信息) [B, C, H+2, W+2]
        padded_feat = F.pad(feat, [1, 1, 1, 1], mode='reflect')

        # 阶段3: 向量化局部特征提取
        # 使用unfold展开为3x3邻域块 [B, C*9, H*W]
        unfolded_feat = F.unfold(padded_feat, kernel_size=3, padding=0)
        # 重塑为 [B, C, 9, H, W] (9对应3x3邻域)
        unfolded_feat = unfolded_feat.view(b, c, 9, h, w)

        # 阶段4: 向量化动态卷积
        # 重塑核参数 [B, 9, 1, 1] → [B, 1, 9, 1, 1]
        kernel_params = kernel_params.view(b, 9)
        kernel_params = kernel_params.view(b, 1, 9, 1, 1)
        # 点乘求和实现卷积 [B, C, H, W]
        gated_feat = (unfolded_feat * kernel_params).sum(dim=2)

        # 阶段5: 应用门控机制
        gate = self.sigmoid(gated_feat)  # [B, C, H, W] ∈ (0,1)
        return feat * gate  # 特征加权输出


# ----------------------------------------------------------#
#   Enhanced Feature Fusion
# ----------------------------------------------------------#
class MultiScaleFusion(nn.Module):
    """增强版特征融合模块

    创新融合策略：
    1. 通道压缩：减少拼接特征的维度
    2. 空间增强：DPConv提取几何感知特征
    3. 特征扩展：恢复通道维度
    4. 残差门控：自适应融合原始信息

    设计优势：
    - 双路径处理：同时保留原始特征和增强特征
    - 空间感知：DPConv建模几何变形
    - 自适应融合：门控机制平衡新旧特征

    Args:
        dim: 特征通道维度
    """

    def __init__(self, dim):
        super().__init__()
        # ------------------------------#
        #   通道压缩层
        # 功能：降维减少计算量
        # 输入：2*dim → 输出：dim
        # ------------------------------#
        self.compress = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),  # 1x1卷积降维
            nn.BatchNorm2d(dim),  # 稳定训练
            nn.ReLU()  # 引入非线性
        )

        # ------------------------------#
        #   空间特征增强
        # 核心：深度可分离卷积增强模块
        # 设计：输出通道减半以降低计算量
        # ------------------------------#
        self.dp_conv = ADPConv(dim, dim // 2)  # [dim → dim//2]

        # ------------------------------#
        #   特征扩展层
        # 功能：恢复通道维度
        # 输入：dim//2 → 输出：dim
        # ------------------------------#
        self.expand = nn.Sequential(
            nn.Conv2d(dim // 2, dim, 1),  # 1x1卷积升维
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # ------------------------------#
        #   残差连接路径
        # 设计：全局上下文压缩
        # 输出：空间不变的特征权重
        # ------------------------------#
        self.residual = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(2 * dim, dim, 1),  # 融合双特征
            nn.Sigmoid()  # 归一化为权重[0,1]
        )

        # ------------------------------#
        #   门控机制
        # 功能：评估增强特征的重要性
        # 输出：通道级权重
        # ------------------------------#
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 空间信息聚合
            nn.Conv2d(dim, dim, 1),  # 特征变换
            nn.Sigmoid()  # 通道权重[0,1]
        )

    def forward(self, feat1, feat2):
        """前向传播：融合双特征

        处理流程：
        1. 特征拼接 → 2. 通道压缩 → 3. 空间增强 → 4. 特征扩展
        5. 残差路径 → 6. 门控生成 → 7. 自适应融合

        融合公式：
        output = residual_weight * gate + enhanced_feature

        维度说明：
        输入: feat1 [B, dim, H, W], feat2 [B, dim, H, W]
        输出: [B, dim, H, W]
        """
        # 阶段1: 特征拼接
        concat = torch.cat([feat1, feat2], dim=1)  # [B, 2*dim, H, W]

        # 阶段2: 通道压缩
        compressed = self.compress(concat)  # [B, dim, H, W]

        # 阶段3: 空间特征增强 (DPConv)
        dp_out = self.dp_conv(compressed)  # [B, dim//2, H, W]

        # 阶段4: 特征扩展
        expanded = self.expand(dp_out)  # [B, dim, H, W]

        # 阶段5: 残差路径 (全局上下文)
        residual = self.residual(concat)  # [B, dim, 1, 1]

        # 阶段6: 门控生成 (评估增强特征重要性)
        gate = self.gate(expanded)  # [B, dim, 1, 1]

        # 阶段7: 自适应融合
        # 公式: output = residual * gate + expanded
        output = residual * gate + expanded  # [B, dim, H, W]

        return output


# ----------------------------------------------------------#
#   Feature Fusion Module (特征融合模块)
# ----------------------------------------------------------#
class BDFM(nn.Module):
    """特征融合模块（整合交叉门控和特征融合）

    三阶段处理流程：
    1. 通道对齐：统一特征维度
    2. 交叉门控：双向特征调制
    3. 增强融合：深度特征整合

    创新特点：
    - 双向门控：特征间相互调制增强
    - 级联融合：门控后深度整合
    - 残差学习：保留原始信息

    Args:
        dim: 特征通道维度
    """

    def __init__(self, dim):
        super().__init__()
        # ------------------------------#
        #   通道对齐层
        # 功能：统一特征维度
        # 设计：1x1卷积保持通道数
        # ------------------------------#
        self.trans_c = nn.Conv2d(dim, dim, 1)  # 通道对齐 (无维度变化)

        # ------------------------------#
        #   交叉门控机制
        # 设计：双向门控增强特征交互
        #  - x_gating: 使用y调制x
        #  - y_gating: 使用x调制y
        # ------------------------------#
        self.x_gating = ContextGating(dim)  # x特征门控 (以y为上下文)
        self.y_gating = ContextGating(dim)  # y特征门控 (以x为上下文)

        # ------------------------------#
        #   增强特征融合
        # 功能：深度整合双路特征
        # 核心：包含空间增强和门控机制
        # ------------------------------#
        self.fusion = MultiScaleFusion(dim)  # 融合门控后特征

        # ------------------------------#
        #   残差连接
        # 设计：恒等映射 (通道数不变)
        # 目的：保留原始特征信息
        # ------------------------------#
        self.res_conv = nn.Identity()  # 无操作残差路径

    def forward(self, x, y):
        """前向传播：融合双特征

        处理流程：
        1. 通道对齐 → 2. 交叉门控 → 3. 特征融合 → 4. 残差连接

        数学表达：
        x' = trans_c(x)
        x_gated = Gating_x(x', y)
        y_gated = Gating_y(y, x')
        fused = Fusion(x_gated, y_gated)
        output = fused + x'

        维度说明：
        输入: x [B, dim, H, W], y [B, dim, H, W]
        输出: [B, dim, H, W] (与x同维度)
        """
        # 阶段1: 通道对齐 (统一特征维度)
        x = self.trans_c(x)  # [B, dim, H, W] → [B, dim, H, W]

        # 阶段2: 交叉门控 (双向特征调制)
        # x特征门控：使用y作为上下文
        x_gated = self.x_gating(x, y)  # [B, dim, H, W]
        # y特征门控：使用x作为上下文
        y_gated = self.y_gating(y, x)  # [B, dim, H, W]

        # 阶段3: 特征融合 (深度整合)
        fused = self.fusion(x_gated, y_gated)  # [B, dim, H, W]

        # 阶段4: 残差连接 (保留原始信息)
        return fused + x  # [B, dim, H, W]

# ----------------------------------------------------------#
#   Atrous Spatial Pyramid Pooling (空洞空间金字塔池化)
# ----------------------------------------------------------#
class DASPP(nn.Module):
    """ASPP模块 - 多尺度特征提取器
    Args:
        dim_in: 输入通道数
        dim_out: 输出通道数（每个分支）
        rate: 基础膨胀率（各分支膨胀率 = rate * 系数）
        bn_mom: BN层的动量参数
    """

    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(DASPP, self).__init__()
        # ------------------------------#
        #   分支1：1x1标准卷积
        # 功能：捕获局部细节特征
        # 膨胀率：1 (无膨胀)
        # ------------------------------#
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # ------------------------------#
        #   分支2：3x3深度可分离卷积
        # 设计：深度卷积+点卷积
        # 膨胀率：rate (基础膨胀)
        # 优势：计算效率高
        # ------------------------------#
        self.branch2 = nn.Sequential(
            ADPConv(dim_in, dim_out, k_size=3),  # 关键修改点
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # ------------------------------#
        #   分支3：5x5深度可分离卷积
        # 膨胀率：rate (中等膨胀)
        # 感受野：中等范围上下文
        # ------------------------------#
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=5, stride=1,
                      padding=2 * rate, dilation=1 * rate, groups=dim_in, bias=False),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # ------------------------------#
        #   分支4：7x7深度可分离卷积
        # 膨胀率：rate (大膨胀)
        # 感受野：大范围上下文
        # ------------------------------#
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=7, stride=1,
                      padding=3 * rate, dilation=1 * rate, groups=dim_in, bias=False),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # ------------------------------#
        #   分支5：全局上下文分支
        # 设计：全局平均池化+卷积
        # 功能：捕获图像级语义信息
        # ------------------------------#
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # ------------------------------#
        #   特征融合层
        # 功能：整合5个分支的特征
        # 设计：1x1卷积压缩通道
        # ------------------------------#
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # ------------------------------#
        #   通道注意力增强
        # 功能：提升特征表达能力
        # 设计：CAB模块自适应特征校准
        # ------------------------------#
        self.cab_attention = MCBAM(in_channels=dim_out)

    def forward(self, x):
        """前向传播流程

        处理步骤：
        1. 并行五分支特征提取
        2. 全局分支上采样恢复尺寸
        3. 特征拼接与融合
        4. 通道注意力增强

        维度说明：
        输入: [B, C, H, W]
        输出: [B, dim_out, H, W]
        """
        b, c, h, w = x.size()  # 获取特征图尺寸

        # ------------------------------#
        #   多尺度特征提取 (并行分支)
        # ------------------------------#
        # 分支1：1x1标准卷积 (局部特征)
        conv1x1 = self.branch1(x)
        # 分支2：3x3深度可分离卷积 (小感受野)
        conv3x3_1 = self.branch2(x)
        # 分支3：5x5深度可分离卷积 (中感受野)
        conv3x3_2 = self.branch3(x)
        # 分支4：7x7深度可分离卷积 (大感受野)
        conv3x3_3 = self.branch4(x)

        # ------------------------------#
        #   全局上下文提取
        # ------------------------------#
        # 全局平均池化 [B, C, 1, 1]
        global_feature = torch.mean(x, dim=[2, 3], keepdim=True)
        # 特征变换 [B, dim_out, 1, 1]
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        # 上采样恢复原始尺寸 [B, dim_out, H, W]
        global_feature = F.interpolate(global_feature, (h, w),
                                       mode='bilinear', align_corners=True)

        # ------------------------------#
        #   特征融合与增强
        # ------------------------------#
        # 拼接多尺度特征 [B, dim_out*5, H, W]
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 1x1卷积融合特征 [B, dim_out, H, W]
        fused = self.conv_cat(feature_cat)
        # 通道注意力增强 [B, dim_out, H, W]
        result = self.cab_attention(fused)

        return result

# ----------------------------------------------------------#
#   DeepLab v3+ 语义分割网络
# ----------------------------------------------------------#
class DeepLab(nn.Module):
    """DeepLab v3+ 主网络

    创新设计：
    1. 多主干支持：MobileNetV2/Xception
    2. 多尺度特征提取：ASPP模块
    3. 特征融合机制：FFM模块整合浅层/深层特征
    4. 自适应下采样：支持8x/16x下采样

    Args:
        num_classes: 分割类别数
        backbone: 主干网络 (mobilenet 或 xception)
        pretrained: 是否使用预训练权重
        downsample_factor: 特征下采样倍数 (16或8)
    """

    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        # ------------------------------#
        #   主干网络初始化
        # 功能：特征提取
        # 输出：浅层特征 + 深层特征
        # ------------------------------#
        if backbone == "xception":
            # Xception主干 (输出两个特征层)
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048  # 深层特征通道数
            low_level_channels = 256  # 浅层特征通道数
        elif backbone == "mobilenet":
            # MobileNetV2主干 (轻量级)
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320  # 深层特征通道数
            low_level_channels = 24  # 浅层特征通道数
        else:
            raise ValueError(f'Unsupported backbone: `{backbone}`. Use mobilenet or xception.')

        # ------------------------------#
        #   ASPP模块 (多尺度特征提取)
        # 输入：主干深层特征
        # 输出：增强的语义特征 (128通道)
        # ------------------------------#
        self.aspp = DASPP(
            dim_in=in_channels,
            dim_out=128,
            rate=16 // downsample_factor  # 自适应膨胀率 (根据下采样倍数调整)
        )

        # ------------------------------#
        #   浅层特征处理分支
        # 功能：通道降维 (匹配ASPP输出)
        # 设计：1x1卷积 + BN + ReLU
        # ------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 128, 1),  # 降维至128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ------------------------------#
        #   特征融合模块
        # 功能：整合浅层细节与深层语义
        # 输入：128通道特征 (双路)
        # ------------------------------#
        self.bdfm = BDFM(dim=128)

        # ------------------------------#
        #   分类头
        # 功能：像素级分类预测
        # 设计：1x1卷积 (无BN/ReLU)
        # ------------------------------#
        self.cls_conv = nn.Conv2d(128, num_classes, 1, stride=1)

    def forward(self, x):
        """前向传播流程

        处理步骤：
        1. 特征提取 → 2. ASPP处理 → 3. 浅层特征处理 →
        4. 特征融合 → 5. 分类预测 → 6. 上采样恢复

        维度变换：
        输入: [B, 3, H, W]
        输出: [B, num_classes, H, W]
        """
        H, W = x.size(2), x.size(3)  # 记录原始输入尺寸

        # =============================#
        #   特征提取阶段
        # =============================#
        # 主干网络提取特征
        # low_level_features: 浅层细节特征 [B, 24/256, H/4, W/4]
        # x: 深层语义特征 [B, 320/2048, H/16, W/16]
        low_level_features, x = self.backbone(x)

        # =============================#
        #   深层特征增强
        # =============================#
        # ASPP处理 (多尺度特征提取)
        # 输入: 深层特征 [B, 320/2048, H/16, W/16]
        # 输出: 增强特征 [B, 128, H/16, W/16]
        x = self.aspp(x)

        # =============================#
        #   浅层特征处理
        # =============================#
        # 降维处理 (匹配融合通道)
        # 输入: [B, 24/256, H/4, W/4]
        # 输出: [B, 128, H/4, W/4]
        low_level_features = self.shortcut_conv(low_level_features)

        # =============================#
        #   特征融合阶段
        # =============================#
        # 上采样深层特征 (匹配浅层尺寸)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, H/8, W/8]
        # 特征融合 (整合浅层细节+深层语义)
        # 输出: [B, 128, H/8, W/8] → [B, 128, H/4, W/4]
        x = self.bdfm(x, low_level_features)

        # =============================#
        #   分类预测阶段
        # =============================#
        # 像素级分类预测
        # 输出: [B, num_classes, H/4, W/4]
        x = self.cls_conv(x)

        # =============================#
        #   输出恢复
        # =============================#
        # 上采样至原始输入尺寸
        # 输出: [B, num_classes, H, W]
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x
