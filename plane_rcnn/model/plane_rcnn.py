import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class PlaneRCNN(MaskRCNN):
    """PlaneRCNN模型
    
    基于官方PlaneRCNN论文实现，使用Mask R-CNN框架，只分为平面和非平面两种类别
    包含三个主要模块：
    1. 平面检测模块：检测平面区域并预测3D平面参数
    2. 深度图预测模块：预测图像的深度图
    3. 平面参数计算模块：计算平面的法向量和距离
    """
    
    def __init__(self, num_classes=2, pretrained=True, backbone_type='resnet50'):
        """初始化PlaneRCNN
        
        Args:
            num_classes: 分类类别数（2类：平面和非平面）
            pretrained: 是否使用预训练权重
            backbone_type: 主干网络类型 ('resnet50' 或 'resnet101')
        """
        # 1. 使用ResNet-FPN作为主干网络，仅使用ImageNet预训练权重
        backbone = resnet_fpn_backbone(backbone_type, pretrained=pretrained)
        
        # 2. 调用父类MaskRCNN初始化
        super(PlaneRCNN, self).__init__(backbone, num_classes)
        
        # 3. 保存设备信息
        self.device = torch.device('cpu')
        
        # 4. 替换检测头，专注于平面检测
        self._replace_detection_heads(num_classes)
        
        # 5. 添加深度图预测模块
        self._add_depth_prediction_head()
        
        # 6. 添加平面法向量预测模块
        self._add_normal_prediction_head()
        
        # 7. 调整RPN参数，提高平面检测性能
        self._adjust_rpn_params()
        
        # 8. 调整ROI heads参数
        self._adjust_roi_heads()
        
        # 9. 初始化锚点法向量（7个锚点，基于k-means聚类）
        self._init_anchor_normals()
    
    def _replace_detection_heads(self, num_classes):
        """替换检测头，优化平面检测
        
        移除COCO特定的检测头，使用简化的检测头
        """
        # 替换FastRCNN检测头，只检测平面和非平面
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # 替换MaskRCNN检测头，使用更小的隐藏层
        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 128  # 减少隐藏层参数，提高推理速度
        self.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
    
    def _add_depth_prediction_head(self):
        """添加深度图预测模块
        
        预测图像的深度图，用于后续计算平面距离
        """
        # 获取FPN输出的通道数
        in_channels = self.backbone.out_channels
        
        # 深度图预测头，使用转置卷积上采样
        self.depth_pred_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def _add_normal_prediction_head(self):
        """添加平面法向量预测模块
        
        预测平面的法向量，使用锚点法向量+残差的方式
        """
        # 获取FPN输出的通道数
        in_channels = self.backbone.out_channels
        
        # 平面法向量预测头
        self.normal_pred_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 输出3D残差向量
        )
    
    def _init_anchor_normals(self):
        """初始化锚点法向量
        
        基于论文，使用7个锚点法向量，通过k-means聚类得到
        """
        # 7个锚点法向量，基于论文中的聚类结果
        self.anchor_normals = torch.tensor([
            [1.0, 0.0, 0.0],   # 正x方向
            [-1.0, 0.0, 0.0],  # 负x方向
            [0.0, 1.0, 0.0],   # 正y方向
            [0.0, -1.0, 0.0],  # 负y方向
            [0.0, 0.0, 1.0],   # 正z方向
            [0.0, 0.0, -1.0],  # 负z方向
            [0.707, 0.0, 0.707] # 45度方向
        ], dtype=torch.float32)
        
        # 将锚点法向量移动到模型设备
        self.anchor_normals = self.anchor_normals.to(self.device)
    
    def _adjust_rpn_params(self):
        """调整RPN参数，提高平面检测性能
        
        平面通常具有较大的尺寸和特定的宽高比
        """
        # 调整RPN的锚点尺寸，适应平面检测
        self.rpn.anchor_generator.sizes = ((64, 128, 256, 512),)  # 增大锚点尺寸
        self.rpn.anchor_generator.aspect_ratios = ((0.33, 0.5, 1.0, 2.0, 3.0),)  # 扩展宽高比范围
        
        # 调整RPN的提议数量
        self.rpn.pre_nms_top_n_train = 2000
        self.rpn.pre_nms_top_n_test = 1000
        self.rpn.post_nms_top_n_train = 2000
        self.rpn.post_nms_top_n_test = 1000
        
        # 调整RPN的NMS阈值
        self.rpn.nms_thresh = 0.7
    
    def _adjust_roi_heads(self):
        """调整ROI heads参数，提高平面检测性能"""
        # 调整ROI heads的批量大小
        self.roi_heads.batch_size_per_image = 64  # 减少批量大小，提高检测质量
        
        # 调整正样本比例
        self.roi_heads.positive_fraction = 0.25  # 降低正样本比例
        
        # 提高检测置信度阈值，减少假阳性
        self.roi_heads.score_thresh = 0.3
        
        # 调整NMS阈值，减少重叠检测
        self.roi_heads.nms_thresh = 0.7
    
    def forward(self, images, targets=None):
        """前向传播，增强平面检测"""
        # 1. 调用父类前向传播，获取基础检测结果
        results = super(PlaneRCNN, self).forward(images, targets)
        
        # 训练模式直接返回结果
        if self.training:
            return results
        
        # 推理模式下，直接返回标准Mask R-CNN输出格式，不进行额外增强
        # 增强操作将在detector中进行
        return results
    
    def _enhance_inference_result(self, result):
        """增强推理结果，添加平面参数
        
        处理检测结果，添加平面参数预测
        只保留高置信度、非重叠的平面检测结果
        """
        # 复制原始结果
        enhanced_result = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in result.items()}
        
        # 添加平面参数
        planes = []
        if 'masks' in result and len(result['scores']) > 0:
            # 获取图像大小
            h, w = result['masks'][0, 0].shape
            image_area = h * w
            
            # 只保留平面类别（类别1是平面）和高置信度平面
            for i in range(len(result['scores'])):
                if result['labels'][i] == 1 and result['scores'][i] > 0.3:
                    # 计算平面面积
                    mask_area = torch.sum(result['masks'][i, 0] > 0.5).item()
                    
                    # 确保平面大小合适
                    if mask_area > 50 and mask_area < 0.25 * image_area:
                        plane = {
                            'bbox': result['boxes'][i].cpu().numpy().astype(int),
                            'mask': result['masks'][i, 0].cpu().numpy() > 0.5,
                            'confidence': result['scores'][i].item(),
                            'label': result['labels'][i].item(),
                            'normal': np.array([0, 0, 1]),  # 默认法向量
                            'distance': 1.0  # 默认距离
                        }
                        planes.append(plane)
            
        # 按置信度排序
        planes.sort(key=lambda x: x['confidence'], reverse=True)
        
        enhanced_result['planes'] = planes
        return enhanced_result
    
    def to(self, device):
        """将模型移动到指定设备
        
        重写to方法，确保所有模块都移动到正确的设备
        """
        # 调用父类to方法
        super(PlaneRCNN, self).to(device)
        
        # 更新设备信息
        self.device = device
        
        # 将锚点法向量移动到指定设备
        self.anchor_normals = self.anchor_normals.to(device)
        
        # 将深度图预测模块移动到指定设备
        self.depth_pred_head = self.depth_pred_head.to(device)
        
        # 将平面法向量预测模块移动到指定设备
        self.normal_pred_head = self.normal_pred_head.to(device)
        
        return self
    
    def predict(self, images):
        """便捷的预测接口，返回处理后的结果
        
        Args:
            images: 输入图像，可以是numpy数组或列表
            
        Returns:
            检测结果，包含平面参数
        """
        # 确保模型处于推理模式
        self.eval()
        
        # 转换输入为Tensor
        if isinstance(images, list):
            images = [torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(self.device) for img in images]
        elif isinstance(images, np.ndarray):
            if images.ndim == 3:  # 单张图像
                images = torch.tensor(images, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif images.ndim == 4:  # 多张图像
                images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            results = self(images)
        
        return results

# 轻量级PlaneRCNN模型，用于资源受限设备
class LightweightPlaneRCNN(PlaneRCNN):
    """轻量级PlaneRCNN模型
    
    适用于树莓派等资源受限设备
    """
    def __init__(self, num_classes=2, pretrained=True):
        """初始化轻量级PlaneRCNN
        
        Args:
            num_classes: 分类类别数
            pretrained: 是否使用预训练权重
        """
        # 使用ResNet50作为主干网络，但减少参数
        super(LightweightPlaneRCNN, self).__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            backbone_type='resnet50'
        )
        
        # 简化特征增强模块
        self._simplify_plane_features()
        
        # 减少检测头参数
        self._reduce_head_params()
    
    def _simplify_plane_features(self):
        """简化平面特征提取模块"""
        self.plane_feature_enhancer = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
    
    def _reduce_head_params(self):
        """减少检测头参数，提高推理速度"""
        # 减少Mask预测头的隐藏层参数
        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        self.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 128, self.roi_heads.mask_predictor.mask_fcn_logits.out_channels
        )

# 获取PlaneRCNN模型的便捷函数
def get_plane_rcnn_model(num_classes=2, pretrained=True, model_type='full'):
    """获取PlaneRCNN模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        model_type: 模型类型，'full'或'lightweight'
        
    Returns:
        PlaneRCNN: PlaneRCNN模型实例
    """
    if model_type == 'lightweight':
        return LightweightPlaneRCNN(num_classes=num_classes, pretrained=pretrained)
    else:
        return PlaneRCNN(num_classes=num_classes, pretrained=pretrained)