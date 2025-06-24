import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

#深度+注意力+融合
class MobileNetV2(nn.Module):
    """适配DeepLab的MobileNetV2改进版
    通过空洞卷积保持特征图分辨率，代替下采样操作
    """

    def __init__(self, downsample_factor=8, pretrained=True):
        """
        Args:
            downsample_factor: 下采样倍数（8或16）
            pretrained: 是否加载预训练权重
        """
        super(MobileNetV2, self).__init__()
        from functools import partial

        # 加载原始MobileNetV2
        model = mobilenetv2(pretrained)
        # 移除最后的分类层前的特征处理（保留主要特征提取层）
        self.features = model.features[:-1]  # 排除最后的1x1卷积和池化

        # 记录关键层索引
        self.total_idx = len(self.features)  # 总层数
        self.down_idx = [2, 4, 7, 14]  # 原始模型中执行下采样的层位置

        # ------------------------------#
        #   根据下采样倍数调整空洞卷积
        # ------------------------------#
        if downsample_factor == 8:
            # 最后两个阶段使用空洞卷积保持分辨率
            # 调整第14层之后的卷积（替换stride=2为空洞卷积）
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)  # 膨胀系数2
                )
            # 调整最后阶段的卷积（更大膨胀系数）
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)  # 膨胀系数4
                )
        elif downsample_factor == 16:
            # 仅调整最后阶段的卷积
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)  # 膨胀系数2
                )

    def _nostride_dilate(self, m, dilate):
        """替换卷积层的stride为dilation（用于保持特征图尺寸）
        Args:
            m: 待修改的卷积层
            dilate: 膨胀系数
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # 处理原始stride=2的卷积层
            if m.stride == (2, 2):
                m.stride = (1, 1)  # 取消下采样
                if m.kernel_size == (3, 3):
                    # 保持感受野：原padding=1 → 新padding=dilate//2
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                # 处理普通卷积层（增加空洞）
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        """前向传播（返回浅层和深层特征）
        输出：
            low_level_features: 浅层特征（用于细节融合）
            x: 深层语义特征
        """
        # 提取前4层的浅层特征（对应原始模型中的早期特征）
        low_level_features = self.features[:4](x)
        # 提取后续层的深层特征
        x = self.features[4:](low_level_features)
        return low_level_features, x


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, min_ratio=8, activation='swish'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.min_ratio = min_ratio

        # 动态计算当前ratio
        self.log_ratio = nn.Parameter(torch.log(torch.tensor(16.0)))  # 初始ratio=16

        # 轻量级通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.SiLU() if activation == 'swish' else nn.ReLU()

        # 深度可分离卷积替代FC层
        current_ratio = max(int(torch.exp(self.log_ratio).item()), self.min_ratio)
        reduced_channels = max(in_channels // current_ratio, 4)  # 防止通道数过小

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=in_channels),
            nn.Conv2d(in_channels, reduced_channels, 1)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(reduced_channels, reduced_channels, 1, groups=reduced_channels),
            nn.Conv2d(reduced_channels, self.out_channels, 1)
        )

        # 空间注意力增强
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 池化融合权重

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 动态计算当前ratio
        current_ratio = max(int(torch.exp(self.log_ratio).item()), self.min_ratio)
        reduced_channels = max(self.in_channels // current_ratio, 4)

        # 通道注意力
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        channel_att = self.sigmoid(self.alpha * avg_out + (1 - self.alpha) * max_out)

        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_att(torch.cat([avg_spatial, max_spatial], dim=1))

        return channel_att * spatial_att  # 联合注意力


class DSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class IDSC(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out


class FFM(nn.Module):
    def __init__(self, dim1):
        super().__init__()
        dim2 = dim1
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)

        self.fusion = nn.Sequential(IDSC(dim2 * 4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())

    def forward(self, x, y):
        b, c, h, w = x.shape
        B, N, C = b, h * w, c
        H = W = h
        x = self.trans_c(x)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)
        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)
        out = torch.cat([x, y, out1, out2], dim=1)
        out = self.fusion(out)
        return out


# -----------------------------------------#
#   ASPP模块：多尺度特征提取器
#   通过不同膨胀率的空洞卷积捕获多尺度上下文
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        """
        Args:
            dim_in: 输入通道数
            dim_out: 输出通道数（每个分支）
            rate: 基础膨胀率（实际膨胀系数为rate的倍数）
            bn_mom: BN层的动量参数
        """
        super(ASPP, self).__init__()
        # 分支1：1x1卷积（捕获局部特征）
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # 分支2：3x3膨胀卷积（rate=6）
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 分支3：3x3膨胀卷积（rate=12）
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1,
                      padding=12 * rate, dilation=12 * rate, groups=dim_in, bias=False),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            CAB(dim_out),
        )

        # 分支4：膨胀率18*rate
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, stride=1,
                      padding=18 * rate, dilation=18 * rate, groups=dim_in, bias=False),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            CAB(dim_out),
        )

        # 分支5：全局平均池化+卷积（捕获全局上下文）
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 更高效的全局池化
            nn.Conv2d(dim_in, dim_out, 1, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
            CAB(dim_out),  # 全局特征增强
        )

        # 特征融合层（整合5个分支）
        self.post_cab = CAB(dim_out * 5, dim_out * 5)  # 输入输出通道保持dim_out*5
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """前向传播流程"""
        [b, c, row, col] = x.size()  # 获取特征图尺寸

        # ------------------------------#
        #   五个分支并行处理
        # ------------------------------#
        # 分支1-4：常规卷积操作
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        # 分支5：全局上下文提取
        global_feat = self.branch5(x)  # [B, d_out, 1, 1]
        global_feat = F.interpolate(global_feat, (row, col), mode='bilinear', align_corners=True)
        # 上采样恢复原始尺寸（双线性插值）

        # ------------------------------#
        #   特征拼接与融合
        # ------------------------------#
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feat], dim=1)
        weighted_feat = self.post_cab(feature_cat) * feature_cat  # 注意力调制
        result = self.conv_cat(weighted_feat)  # 通道压缩
        return result

class DeepLab(nn.Module):
    """DeepLab v3+ 主网络"""

    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        """
        Args:
            num_classes: 分类类别数
            backbone: 主干网络（mobilenet或xception）
            pretrained: 是否使用预训练权重
            downsample_factor: 特征下采样倍数（16或8）
        """
        super(DeepLab, self).__init__()
        # ------------------------------#
        #   主干网络选择与初始化
        # ------------------------------#
        if backbone == "xception":
            # Xception结构（输出两个特征层）
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048  # 主干特征通道数
            low_level_channels = 256  # 浅层特征通道数
        elif backbone == "mobilenet":
            # MobileNetV2结构（输出两个特征层）
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320  # 主干特征通道数
            low_level_channels = 24  # 浅层特征通道数
        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`, Use mobilenet, xception.')

        # ------------------------------#
        #   ASPP模块构建
        # ------------------------------#
        self.aspp = ASPP(
            dim_in=in_channels,
            dim_out=256,
            rate=16 // downsample_factor  # 基础膨胀率计算（根据下采样倍数调整）
        )

        # ------------------------------#
        #   浅层特征处理分支
        # ------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 256, 1),  # 1x1卷积降维
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ------------------------------#
        #   特征融合与分类头
        # ------------------------------#
        self.ffm = FFM(dim1=256)
        # 最终分类卷积（输出类别数）
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        """前向传播流程"""
        H, W = x.size(2), x.size(3)  # 记录输入图像原始尺寸

        # ------------------------------#
        #   特征提取
        # ------------------------------#
        # 获取浅层特征（用于细节恢复）和主干特征
        low_level_features, x = self.backbone(x)
        # ASPP处理主干特征（增强语义信息）
        x = self.aspp(x)
        # 处理浅层特征（降维匹配）
        low_level_features = self.shortcut_conv(low_level_features)

        # ------------------------------#
        #   特征融合
        # ------------------------------#
        # 将主干特征上采样至浅层特征尺寸
        x = F.interpolate(x,
                          size=(low_level_features.size(2), low_level_features.size(3)),
                          mode='bilinear',
                          align_corners=True)
        # 拼接特征并融合
        x = self.ffm(x, low_level_features)

        # ------------------------------#
        #   最终分类与上采样
        # ------------------------------#
        x = self.cls_conv(x)  # 分类卷积
        # 上采样回原始输入尺寸
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
