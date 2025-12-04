import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightFPN(nn.Module):
    """轻量级特征金字塔网络(Feature Pyramid Network)
    
    实现了自下而上、自上而下和横向连接的特征融合，生成多尺度特征金字塔
    针对边缘设备优化，减少了通道数
    
    Args:
        in_channels_list: 输入特征图的通道数列表
        out_channels: 输出特征图的通道数，默认64
    """
    def __init__(self, in_channels_list, out_channels=64):
        super(LightweightFPN, self).__init__()
        
        # 输出通道数，默认64，适合边缘设备
        self.out_channels = out_channels
        
        # 横向连接卷积层：将不同尺度的特征图转换为相同通道数
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
        
        # 输出卷积层：对融合后的特征图进行卷积，生成最终的特征金字塔
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        """前向传播
        
        Args:
            inputs: 自下而上路径的特征图列表，从低层到高层
            
        Returns:
            list: 特征金字塔，从高层到低层
        """
        assert len(inputs) == len(self.lateral_convs), "输入特征图数量与横向卷积层数量不匹配"
        
        # 1. 横向连接：使用1x1卷积将不同尺度的特征图转换为相同通道数
        lateral_features = []
        for i in range(len(inputs)):
            lateral_features.append(self.lateral_convs[i](inputs[i]))
        
        # 2. 自上而下路径：将高层特征通过上采样与低层特征融合
        # 从最高层开始
        pyramid_features = [lateral_features[-1]]
        
        # 从次高层到最低层
        for i in range(len(lateral_features)-2, -1, -1):
            # 上采样：使用最近邻插值将高层特征图放大到与当前层相同尺寸
            upsampled = F.interpolate(
                pyramid_features[-1], 
                size=lateral_features[i].shape[2:], 
                mode='nearest'
            )
            
            # 特征融合：当前层横向连接特征 + 上采样后的高层特征
            fused = lateral_features[i] + upsampled
            pyramid_features.append(fused)
        
        # 反转列表，使特征金字塔从低层到高层
        pyramid_features.reverse()
        
        # 3. 输出卷积：对融合后的特征图进行3x3卷积，生成最终的特征金字塔
        for i in range(len(pyramid_features)):
            pyramid_features[i] = self.output_convs[i](pyramid_features[i])
        
        return pyramid_features

# 简化版FPN，用于极小模型
class UltraLightweightFPN(nn.Module):
    """超轻量级特征金字塔网络
    
    进一步简化了FPN结构，仅保留必要的特征融合
    
    Args:
        in_channels_list: 输入特征图的通道数列表
        out_channels: 输出特征图的通道数，默认32
    """
    def __init__(self, in_channels_list, out_channels=32):
        super(UltraLightweightFPN, self).__init__()
        self.out_channels = out_channels
        
        # 仅保留最主要的特征融合
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list[-2:]:  # 仅使用最后两层特征
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
        
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs):
        """前向传播
        
        Args:
            inputs: 自下而上路径的特征图列表，从低层到高层
            
        Returns:
            list: 简化的特征金字塔
        """
        # 仅使用最后两层特征
        inputs = inputs[-2:]
        lateral_features = [self.lateral_convs[i](inputs[i]) for i in range(len(inputs))]
        
        # 特征融合
        upsampled = F.interpolate(
            lateral_features[-1], 
            size=lateral_features[0].shape[2:], 
            mode='nearest'
        )
        fused = lateral_features[0] + upsampled
        
        # 输出卷积
        output = self.output_conv(fused)
        
        return [output]  # 仅返回一个融合后的特征图

if __name__ == "__main__":
    # 测试轻量级FPN
    import torch
    
    # 模拟输入特征
    in_channels_list = [64, 128, 256, 512]  # ResNet-18各层输出通道
    inputs = [
        torch.randn(1, 64, 56, 56),
        torch.randn(1, 128, 28, 28),
        torch.randn(1, 256, 14, 14),
        torch.randn(1, 512, 7, 7)
    ]
    
    # 创建轻量级FPN
    fpn = LightweightFPN(in_channels_list, out_channels=64)
    
    # 生成特征金字塔
    pyramid = fpn(inputs)
    
    print("轻量级FPN测试:")
    print(f"输入特征数量: {len(inputs)}")
    print(f"输出特征数量: {len(pyramid)}")
    for i, p in enumerate(pyramid):
        print(f"特征{i}尺寸: {p.shape}")
    
    # 测试超轻量级FPN
    ultra_fpn = UltraLightweightFPN(in_channels_list, out_channels=32)
    ultra_pyramid = ultra_fpn(inputs)
    
    print("\n超轻量级FPN测试:")
    print(f"输入特征数量: {len(inputs)}")
    print(f"输出特征数量: {len(ultra_pyramid)}")
    for i, p in enumerate(ultra_pyramid):
        print(f"特征{i}尺寸: {p.shape}")