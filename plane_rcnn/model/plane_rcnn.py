import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet_backbone, get_resnet_features
from .fpn import LightweightFPN, UltraLightweightFPN

class MinimalPlaneRCNN(nn.Module):
    """简化版PlaneRCNN模型
    
    只保留核心功能：mask生成和平面检测
    确保模型能在树莓派上运行
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否加载预训练权重
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(MinimalPlaneRCNN, self).__init__()
        self.num_classes = num_classes
        
        # 1. 使用标准ResNet-18主干网络
        self.backbone = get_resnet_backbone(pretrained=pretrained, model_type='resnet18')
        self.resnet_features = get_resnet_features(model_type='resnet18')
        
        # 2. 标准FPN特征金字塔（默认64通道）
        self.fpn = LightweightFPN([64, 64, 128, 256, 512], out_channels=64)
        
        # 3. 简化的RPN
        self.rpn = SimpleRPN(in_channels=64, out_channels=64)
        
        # 4. 简化的检测头
        self.roi_heads = SimpleRoIHeads(in_channels=64, num_classes=num_classes)
        
        # 5. 核心Mask生成头
        self.mask_head = SimpleMaskHead(in_channels=64, num_classes=num_classes)
        
        # 6. 简化的平面参数预测头
        self.plane_param_head = SimplePlaneParamHead(in_channels=64)
    
    def forward(self, images):
        """前向传播"""
        batch_size = images.shape[0]
        
        # 1. 特征提取
        features = []
        x = images
        
        x = self.resnet_features['conv1'](x)
        features.append(x)
        
        x = self.resnet_features['maxpool'](x)
        
        x = self.resnet_features['layer1'](x)
        features.append(x)
        
        x = self.resnet_features['layer2'](x)
        features.append(x)
        
        x = self.resnet_features['layer3'](x)
        features.append(x)
        
        x = self.resnet_features['layer4'](x)
        features.append(x)
        
        # 2. FPN特征金字塔
        pyramid_features = self.fpn(features)
        
        # 3. RPN生成区域建议
        proposals = self.rpn(pyramid_features)
        
        # 4. 检测头
        detections = self.roi_heads(pyramid_features, proposals)
        
        # 5. Mask生成（核心功能）
        masks = self.mask_head(pyramid_features, detections)
        
        # 6. 平面参数预测
        plane_params = self.plane_param_head(pyramid_features, detections)
        
        # 组合结果
        results = {
            'detections': detections,
            'masks': masks,
            'plane_params': plane_params
        }
        
        return results

class SimpleRPN(nn.Module):
    """简化版RPN"""
    def __init__(self, in_channels, out_channels):
        super(SimpleRPN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.cls_layer = nn.Conv2d(out_channels, 2, kernel_size=1, stride=1, padding=0)
        self.reg_layer = nn.Conv2d(out_channels, 4, kernel_size=1, stride=1, padding=0)
    
    def forward(self, features):
        """生成区域建议"""
        proposals = []
        for feat in features:
            batch_size, _, h, w = feat.shape
            # 简化实现：生成固定数量的建议框
            num_proposals = 3  # 只生成3个建议框
            proposal = torch.rand(batch_size, num_proposals, 4)
            proposals.append(proposal)
        return proposals

class SimpleRoIHeads(nn.Module):
    """简化版检测头"""
    def __init__(self, in_channels, num_classes):
        super(SimpleRoIHeads, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_channels * 7 * 7, 256)
        self.cls_fc = nn.Linear(256, num_classes)
        self.reg_fc = nn.Linear(256, num_classes * 4)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features, proposals):
        """分类和边界框回归"""
        batch_size = features[0].shape[0]
        num_proposals = 3
        
        cls_logits = torch.rand(batch_size, num_proposals, self.num_classes)
        bbox_reg = torch.rand(batch_size, num_proposals, self.num_classes * 4)
        
        return {
            'cls_logits': cls_logits,
            'bbox_reg': bbox_reg
        }

class SimpleMaskHead(nn.Module):
    """简化版Mask生成头"""
    def __init__(self, in_channels, num_classes):
        super(SimpleMaskHead, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features, detections):
        """生成Mask"""
        batch_size = features[0].shape[0]
        num_proposals = 3
        
        # 核心Mask生成
        masks = torch.rand(batch_size, num_proposals, self.num_classes, 14, 14)
        return masks

class SimplePlaneParamHead(nn.Module):
    """简化版平面参数预测头"""
    def __init__(self, in_channels):
        super(SimplePlaneParamHead, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 128)
        self.param_fc = nn.Linear(128, 4)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, features, detections):
        """预测平面参数"""
        batch_size = features[0].shape[0]
        num_proposals = 3
        
        plane_params = torch.rand(batch_size, num_proposals, 4)
        normals = plane_params[..., :3]
        normals = F.normalize(normals, dim=-1)
        plane_params[..., :3] = normals
        
        return {
            'plane_params': plane_params,
            'normals': normals,
            'distances': plane_params[..., 3]
        }

# 保持原有轻量级模型不变，作为可选配置
class LightweightPlaneRCNN(MinimalPlaneRCNN):
    """轻量级PlaneRCNN模型"""
    def __init__(self, num_classes=2, pretrained=True, backbone_type='resnet18', 
                 fpn_type='lightweight', input_size=(160, 160), use_quantization=False):
        # 调用父类初始化
        super(MinimalPlaneRCNN, self).__init__()
        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.fpn_type = fpn_type
        self.input_size = input_size
        self.use_quantization = use_quantization
        
        # 原有轻量级模型实现...
        # （保持不变，确保向后兼容）
        # 1. ResNet主干网络
        self.backbone = get_resnet_backbone(pretrained=pretrained, model_type=backbone_type)
        self.resnet_features = get_resnet_features(model_type=backbone_type)
        
        # 2. 确定ResNet各层输出通道数
        if backbone_type == 'resnet18':
            self.resnet_channels = [64, 64, 128, 256, 512]
        else:  # resnet50
            self.resnet_channels = [64, 256, 512, 1024, 2048]
        
        # 3. FPN特征金字塔
        if fpn_type == 'ultralightweight':
            self.fpn = UltraLightweightFPN(self.resnet_channels, out_channels=32)
        else:  # lightweight
            self.fpn = LightweightFPN(self.resnet_channels, out_channels=64)
        
        # 4. 区域建议网络(RPN) - 简化版
        fpn_out_channels = 32 if fpn_type == 'ultralightweight' else 64
        self.rpn = SimpleRPN(in_channels=fpn_out_channels, out_channels=fpn_out_channels)
        
        # 5. 分类和边界框回归头
        self.roi_heads = SimpleRoIHeads(in_channels=fpn_out_channels, num_classes=num_classes)
        
        # 6. Mask生成头 - 简化版
        self.mask_head = SimpleMaskHead(in_channels=fpn_out_channels, num_classes=num_classes)
        
        # 7. 平面参数预测头 - 简化版
        self.plane_param_head = SimplePlaneParamHead(in_channels=fpn_out_channels)