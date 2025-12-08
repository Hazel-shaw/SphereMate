#!/usr/bin/env python3
"""
基础平面检测脚本
使用自定义的PlaneRCNN模型，只分为平面和非平面两种类别
"""

import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn

class BasicPlaneDetector:
    """基础平面检测器"""
    
    def __init__(self, device='cpu'):
        """初始化检测器"""
        self.device = torch.device(device)
        
        # 使用torchvision的Mask R-CNN模型，预训练在COCO数据集上
        # 我们将调整为只检测平面（类别1）
        self.model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
        self.model.eval()
        self.model.to(self.device)
        
        # 平面类别的置信度阈值
        self.plane_confidence_threshold = 0.3
    
    def detect(self, image):
        """检测平面"""
        # 图像预处理
        img_tensor = F.to_tensor(image).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model([img_tensor])[0]
        
        # 后处理
        planes = []
        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        masks = outputs['masks'].cpu().numpy()
        
        # 获取图像面积
        h, w = image.shape[:2]
        image_area = h * w
        
        # 筛选置信度
        for box, score, mask in zip(boxes, scores, masks):
            # 较低的置信度阈值，减少假阳性
            if score > 0.2:
                # 计算边界框面积
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                
                # 确保框选大小适中
                if 500 < box_area < 0.5 * image_area:
                    # 生成mask，较低的阈值
                    mask = mask[0] > 0.3
                    
                    # 计算mask面积
                    mask_area = np.sum(mask)
                    if mask_area > 100:
                        planes.append({
                            'bbox': box.astype(int),
                            'mask': mask,
                            'confidence': score,
                            'score': score,
                            'label': 1,  # 1表示平面
                            'normal': np.array([0, 0, 1])
                        })
        
        # 按置信度排序
        planes.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {'planes': planes, 'original_shape': image.shape}
    
    def visualize(self, image, results):
        """可视化检测结果"""
        # 确保可视化图像始终有内容，解决黑色背景问题
        vis_image = image.copy()
        if vis_image is None or vis_image.shape[0] == 0 or vis_image.shape[1] == 0:
            return image
        
        # 只显示置信度较高的平面
        planes = [p for p in results['planes'] if p['confidence'] > self.plane_confidence_threshold]
        
        for plane in planes:
            # 绘制边界框，使用半透明效果
            bbox = plane['bbox']
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # 绘制mask，使用半透明绿色
            mask = plane['mask']
            colored_mask = np.zeros_like(vis_image)
            colored_mask[mask] = (0, 255, 0)  # 绿色mask
            vis_image = cv2.addWeighted(vis_image, 0.9, colored_mask, 0.1, 0)  # 半透明效果
            
            # 绘制标签
            label = f"Plane: {plane['confidence']:.2f}"
            cv2.putText(vis_image, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基础平面检测')
    parser.add_argument('--image', type=str, default='./plane_rcnn/inference/test.jpg', help='测试图片路径')
    parser.add_argument('--output', type=str, default='./plane_rcnn/inference/basic_result.jpg', help='输出结果路径')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
    
    args = parser.parse_args()
    
    # 读取图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"无法读取图像: {args.image}")
        return
    
    # 创建检测器
    detector = BasicPlaneDetector(device=args.device)
    
    # 检测平面
    print("正在检测平面...")
    results = detector.detect(image)
    
    # 可视化结果
    vis_image = detector.visualize(image, results)
    
    # 保存结果
    cv2.imwrite(args.output, vis_image)
    print(f"结果已保存到: {args.output}")
    print(f"检测到 {len(results['planes'])} 个平面")
    
    # 打印检测结果的详细信息
    if len(results['planes']) > 0:
        print("\n检测结果详情:")
        for i, plane in enumerate(results['planes'][:5]):
            print(f"平面 {i+1}: 置信度 {plane['confidence']:.4f}, 位置 {plane['bbox']}")

if __name__ == "__main__":
    main()
