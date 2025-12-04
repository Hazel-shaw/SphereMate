#!/usr/bin/env python3
"""
SphereMate主应用入口

用于启动PlaneRCNN平面检测与投影系统
默认使用简化版模型，确保在树莓派上运行
"""

import argparse
import cv2
import numpy as np
from plane_rcnn.utils.camera import Camera
from plane_rcnn.inference.detector import PlaneDetector
from plane_rcnn.projection.projector import Projector
from plane_rcnn.projection.visualizer import PlaneVisualizer
from plane_rcnn.projection.calibration import CameraProjectorCalibrator

class SphereMateApp:
    """SphereMate应用主类"""
    def __init__(self, config=None):
        """初始化应用"""
        self.config = config if config is not None else {}
        self.is_running = False
        
        # 组件初始化
        self.camera = None
        self.detector = None
        self.projector = None
        self.visualizer = None
        self.calibrator = None
    
    def initialize(self):
        """初始化应用组件"""
        print("正在初始化SphereMate应用...")
        
        # 1. 初始化摄像头
        self.camera = Camera(config=self.config.get('camera', {}))
        if not self.camera.open():
            print("摄像头初始化失败")
            return False
        
        # 2. 初始化平面检测器 - 默认使用简化版模型
        detector_config = self.config.get('detector', {})
        self.detector = PlaneDetector(**detector_config)
        
        # 3. 初始化投影系统
        self.projector = Projector(config=self.config.get('projector', {}))
        if not self.projector.connect():
            print("投影系统初始化失败")
            # 投影系统失败不影响应用启动
        
        # 4. 初始化平面可视化器
        self.visualizer = PlaneVisualizer(config=self.config.get('visualizer', {}))
        
        # 5. 初始化标定器
        self.calibrator = CameraProjectorCalibrator(config=self.config.get('calibration', {}))
        
        self.is_running = True
        print("SphereMate应用初始化成功")
        return True
    
    def run_image_detection(self, image_path, output_path=None):
        """运行图像平面检测"""
        print(f"正在处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 平面检测
        results = self.detector.detect(image)
        
        # 可视化结果
        vis_image = self.visualizer.visualize_planes(image, results)
        
        # 显示结果
        cv2.imshow('Plane Detection Result', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if output_path is not None:
            cv2.imwrite(output_path, vis_image)
            print(f"结果已保存到: {output_path}")
    
    def run_realtime_detection(self):
        """运行实时平面检测"""
        print("开始实时平面检测...")
        print("按ESC键退出")
        
        frame_count = 0
        
        while self.is_running:
            # 捕获图像
            frame = self.camera.capture_frame()
            
            if frame is not None:
                frame_count += 1
                
                # 平面检测
                results = self.detector.detect(frame)
                
                # 可视化结果
                vis_frame = self.visualizer.visualize_planes(frame, results)
                
                # 添加FPS信息
                fps = f"FPS: {frame_count}"
                cv2.putText(vis_frame, fps, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('Real-time Plane Detection', vis_frame)
                
                # 生成投影图像
                if self.projector.is_connected:
                    proj_image = self.visualizer.generate_projection_image(
                        results, 
                        self.config.get('projection_matrix', np.eye(3)),
                        self.projector.projector_resolution
                    )
                    self.projector.project_image(proj_image)
            
            # 检测按键
            key = cv2.waitKey(1)
            if key == 27:  # ESC键
                self.is_running = False
        
        print("实时平面检测已停止")
    
    def shutdown(self):
        """关闭应用"""
        print("正在关闭SphereMate应用...")
        
        # 关闭摄像头
        if self.camera is not None:
            self.camera.close()
        
        # 关闭投影系统
        if self.projector is not None:
            self.projector.disconnect()
        
        # 关闭所有窗口
        cv2.destroyAllWindows()
        
        self.is_running = False
        print("SphereMate应用已关闭")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SphereMate PlaneRCNN应用')
    parser.add_argument('--mode', type=str, choices=['image', 'realtime', 'mac_camera'], 
                      default='realtime', help='运行模式')
    parser.add_argument('--image', type=str, help='输入图像路径（仅image模式）')
    parser.add_argument('--output', type=str, help='输出结果路径（仅image模式）')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
    parser.add_argument('--use-lightweight', action='store_true', 
                      help='使用轻量级模型')
    parser.add_argument('--input-size', type=int, nargs=2, default=(160, 160), 
                      help='模型输入尺寸')
    
    args = parser.parse_args()
    
    # 初始化应用配置
    config = {
        'detector': {
            'device': args.device,
            'use_lightweight': args.use_lightweight,
            'lightweight_config': {
                'backbone_type': 'resnet18',
                'fpn_type': 'lightweight',
                'input_size': tuple(args.input_size),
                'use_quantization': False
            }
        },
        'camera': {
            'resolution': (640, 480),
            'fps': 30
        },
        'projector': {
            'resolution': (1920, 1080),
            'brightness': 50
        }
    }
    
    if args.mode == 'mac_camera':
        # 使用MacBook Pro摄像头测试代码
        print("启动MacBook Pro摄像头平面检测...")
        from mac_camera_test import MacCameraPlaneDetector
        
        # 创建Mac摄像头检测器
        mac_detector = MacCameraPlaneDetector(
            camera_id=0,
            resolution=(640, 480),
            model_config={
                'device': args.device,
                'use_lightweight': args.use_lightweight,
                'lightweight_config': config['detector']['lightweight_config']
            }
        )
        
        # 运行实时检测
        mac_detector.run_realtime_detection()
        return 0
    
    # 创建应用实例
    app = SphereMateApp(config=config)
    
    # 初始化应用
    if not app.initialize():
        print("应用初始化失败")
        return 1
    
    try:
        # 根据模式运行应用
        if args.mode == 'image':
            if args.image is None:
                print("错误：image模式需要指定--image参数")
                return 1
            app.run_image_detection(args.image, args.output)
        elif args.mode == 'realtime':
            app.run_realtime_detection()
        
    except KeyboardInterrupt:
        print("\n应用被用户中断")
    except Exception as e:
        print(f"应用运行出错: {e}")
    finally:
        # 关闭应用
        app.shutdown()
    
    return 0

if __name__ == "__main__":
    exit(main())