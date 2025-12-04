#!/usr/bin/env python3
"""
MacBook Pro摄像头实时平面检测

使用简化版PlaneRCNN模型，在MacBook Pro的摄像头实时流中检测平面
"""

import cv2
import numpy as np
from plane_rcnn.inference.detector import PlaneDetector

class MacCameraPlaneDetector:
    """MacBook Pro摄像头平面检测器"""
    def __init__(self, camera_id=0, resolution=(640, 480), 
                 model_config=None):
        """初始化检测器
        
        Args:
            camera_id: 摄像头ID，默认0
            resolution: 摄像头分辨率，默认(640, 480)
            model_config: 模型配置字典
        """
        # 摄像头配置
        self.camera_id = camera_id
        self.resolution = resolution
        
        # 模型配置
        if model_config is None:
            model_config = {
                'device': 'cpu',
                'use_lightweight': False,  # 默认使用简化版模型
                'lightweight_config': {
                    'backbone_type': 'resnet18',
                    'fpn_type': 'lightweight',
                    'input_size': (160, 160),
                    'use_quantization': False
                }
            }
        
        # 初始化平面检测器
        self.detector = PlaneDetector(**model_config)
        
        # 初始化摄像头
        self.cap = None
    
    def initialize_camera(self):
        """初始化摄像头"""
        print(f"正在初始化摄像头 {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return False
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        print(f"摄像头 {self.camera_id} 初始化成功，分辨率: {self.resolution}")
        return True
    
    def run_realtime_detection(self):
        """运行实时平面检测"""
        if not self.cap:
            if not self.initialize_camera():
                return False
        
        print("开始实时平面检测...")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        
        frame_count = 0
        
        while True:
            # 捕获图像
            ret, frame = self.cap.read()
            
            if not ret:
                print("无法获取图像帧")
                break
            
            frame_count += 1
            
            # 进行平面检测
            results = self.detector.detect(frame)
            
            # 可视化检测结果
            vis_frame = self.detector.visualize_detections(frame, results)
            
            # 添加FPS信息
            fps = f"FPS: {frame_count}"  # 简化的FPS计算
            cv2.putText(vis_frame, fps, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow("MacBook Pro 实时平面检测", vis_frame)
            
            # 处理按键
            key = cv2.waitKey(1)
            
            if key == ord('q'):  # 退出
                print("退出实时检测")
                break
            elif key == ord('s'):  # 保存当前帧
                save_path = f"plane_detection_frame_{frame_count}.jpg"
                cv2.imwrite(save_path, vis_frame)
                print(f"帧已保存到: {save_path}")
        
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()
        
        return True
    
    def run_image_detection(self, image_path, output_path=None):
        """运行图像平面检测
        
        Args:
            image_path: 输入图像路径
            output_path: 输出结果路径
            
        Returns:
            bool: 检测成功返回True，否则返回False
        """
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图像: {image_path}")
            return False
        
        # 进行平面检测
        results = self.detector.detect(image)
        
        # 可视化检测结果
        vis_image = self.detector.visualize_detections(image, results)
        
        # 显示结果
        cv2.imshow("图像平面检测", vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"检测结果已保存到: {output_path}")
        
        return True
    
    def __del__(self):
        """析构函数，释放资源"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    """主函数"""
    # 1. 创建MacBook Pro摄像头平面检测器
    detector = MacCameraPlaneDetector(
        camera_id=0,  # 默认摄像头
        resolution=(640, 480),  # 降低分辨率以提高帧率
        model_config={
            'device': 'cpu',
            'use_lightweight': False  # 默认使用简化版模型
        }
    )
    
    # 2. 运行实时检测
    detector.run_realtime_detection()

if __name__ == "__main__":
    main()