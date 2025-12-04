import torch
import cv2
import numpy as np
from plane_rcnn.model.plane_rcnn import MinimalPlaneRCNN, LightweightPlaneRCNN

class PlaneDetector:
    """平面检测器
    
    用于图像平面检测的推理接口，默认使用简化版模型确保在树莓派上运行
    """
    def __init__(self, model_path=None, num_classes=2, device='cpu', 
                 use_lightweight=False, lightweight_config=None):
        """初始化平面检测器
        
        Args:
            model_path: 模型权重路径
            num_classes: 分类类别数
            device: 运行设备('cpu'或'cuda')
            use_lightweight: 是否使用轻量级模型
            lightweight_config: 轻量级模型配置
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.use_lightweight = use_lightweight
        
        # 1. 创建模型 - 默认使用简化版模型
        if use_lightweight:
            # 使用轻量级模型
            if lightweight_config is None:
                lightweight_config = {
                    'backbone_type': 'resnet18',
                    'fpn_type': 'lightweight',
                    'input_size': (160, 160),
                    'use_quantization': False
                }
            self.model = LightweightPlaneRCNN(
                num_classes=num_classes,
                pretrained=True,
                **lightweight_config
            )
            self.input_size = lightweight_config['input_size']
        else:
            # 使用简化版模型（默认）
            self.model = MinimalPlaneRCNN(
                num_classes=num_classes,
                pretrained=True
            )
            self.input_size = (160, 160)  # 默认输入尺寸
        
        # 2. 加载模型权重
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功加载模型权重: {model_path}")
            except Exception as e:
                print(f"加载模型权重失败: {e}")
                print("将使用随机初始化权重")
        
        # 3. 设置模型为推理模式
        self.model.eval()
        self.model.to(self.device)
        
        # 4. 图像预处理参数
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def preprocess(self, image):
        """图像预处理"""
        # 1. 转换为RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 调整图像尺寸
        resized = cv2.resize(image_rgb, self.input_size)
        
        # 3. 归一化到[0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # 4. 减去均值，除以标准差
        normalized = (normalized - self.mean) / self.std
        
        # 5. 转换为CHW格式
        chw = np.transpose(normalized, (2, 0, 1))
        
        # 6. 转换为Tensor
        tensor = torch.from_numpy(chw)
        
        # 7. 添加batch维度
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def detect(self, image):
        """检测图像中的平面"""
        # 1. 图像预处理
        input_tensor = self.preprocess(image)
        
        # 2. 模型推理
        with torch.no_grad():
            results = self.model(input_tensor)
        
        # 3. 后处理
        processed_results = self.post_process(results, image.shape)
        
        return processed_results
    
    def post_process(self, results, original_shape):
        """结果后处理"""
        # 1. 分类结果处理
        cls_logits = results['detections']['cls_logits'][0]  # (N, num_classes)
        cls_scores = torch.softmax(cls_logits, dim=-1)  # 转换为概率
        
        # 2. 边界框处理
        bbox_reg = results['detections']['bbox_reg'][0]  # (N, num_classes*4)
        
        # 3. Mask处理（核心功能）
        masks = results['masks'][0]  # (N, num_classes, 14, 14)
        masks = torch.sigmoid(masks) > 0.5  # 转换为二值mask
        
        # 4. 平面参数处理
        plane_params = results['plane_params']['plane_params'][0]  # (N, 4)
        normals = results['plane_params']['normals'][0]  # (N, 3)
        distances = results['plane_params']['distances'][0]  # (N,)
        
        # 5. 筛选置信度高的检测结果
        plane_indices = self._filter_detections(cls_scores)
        
        # 6. 调整边界框和mask到原始图像尺寸
        original_h, original_w = original_shape[:2]
        
        # 转换mask尺寸
        masks_resized = []
        for i in plane_indices:
            mask = masks[i, 1].cpu().numpy()  # 类别1的mask
            mask_resized = cv2.resize(mask.astype(np.float32), (original_w, original_h))
            masks_resized.append(mask_resized > 0.5)
        
        masks_resized = np.array(masks_resized)
        
        # 简化的边界框计算（基于mask）
        bboxes = []
        for mask in masks_resized:
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue
            
            x1 = np.min(coords[1])
            y1 = np.min(coords[0])
            x2 = np.max(coords[1])
            y2 = np.max(coords[0])
            bboxes.append([x1, y1, x2, y2])
        
        bboxes = np.array(bboxes)
        
        # 7. 组织结果
        processed_results = {
            'planes': [],
            'original_shape': original_shape,
            'input_size': self.input_size
        }
        
        for i, plane_idx in enumerate(plane_indices):
            if i >= len(bboxes) or i >= len(masks_resized):
                continue
            
            plane = {
                'bbox': bboxes[i],
                'mask': masks_resized[i],
                'confidence': cls_scores[plane_idx, 1].item(),
                'normal': normals[plane_idx].cpu().numpy(),
                'distance': distances[plane_idx].item(),
                'params': plane_params[plane_idx].cpu().numpy()
            }
            
            processed_results['planes'].append(plane)
        
        return processed_results
    
    def _filter_detections(self, cls_scores, confidence_threshold=0.3, max_detections=3):
        """筛选检测结果"""
        # 获取平面类别的置信度（类别1）
        plane_scores = cls_scores[:, 1]
        
        # 筛选置信度大于阈值的检测结果
        indices = torch.where(plane_scores > confidence_threshold)[0]
        
        # 按置信度排序
        sorted_indices = indices[torch.argsort(plane_scores[indices], descending=True)]
        
        # 取前max_detections个结果
        return sorted_indices[:max_detections].tolist()
    
    def visualize_detections(self, image, results):
        """可视化检测结果"""
        # 复制图像以避免修改原始图像
        vis_image = image.copy()
        
        # 遍历检测到的平面
        for i, plane in enumerate(results['planes']):
            # 1. 获取平面信息
            bbox = plane['bbox'].astype(int)
            mask = plane['mask'].astype(np.uint8)
            confidence = plane['confidence']
            
            # 2. 绘制边界框
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 3. 绘制Mask（核心功能）
            # 创建彩色mask
            color = np.array([0, 0, 255], dtype=np.uint8)  # 红色
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask > 0] = color
            
            # 融合mask到原始图像
            vis_image = cv2.addWeighted(vis_image, 0.8, colored_mask, 0.2, 0)
            
            # 4. 添加文字标注
            label = f"Plane {i+1}: {confidence:.2f}"
            cv2.putText(vis_image, label, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image

def main():
    """示例用法"""
    # 1. 创建平面检测器 - 默认使用简化版模型
    detector = PlaneDetector(
        device='cpu',
        use_lightweight=False  # 默认使用简化版模型
    )
    
    # 2. 读取测试图像
    test_image_path = "../ResNet_project/data/fruits-360_multi/test-multiple_fruits/apple.jpg"
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"无法读取测试图像: {test_image_path}")
        return
    
    # 3. 进行平面检测
    print("开始平面检测...")
    results = detector.detect(image)
    
    # 4. 可视化检测结果
    print(f"检测到 {len(results['planes'])} 个平面")
    vis_image = detector.visualize_detections(image, results)
    
    # 5. 保存可视化结果
    output_path = "test_minimal_plane_detection.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"可视化结果已保存到: {output_path}")

if __name__ == "__main__":
    main()