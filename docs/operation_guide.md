# PlaneRCNN模块操作指南

## 1. 概述

PlaneRCNN模块用于单图像3D平面检测与重建，并将检测结果投影到物理空间。本指南将详细介绍该模块的安装、配置和使用方法。

## 2. 目录结构

```
plane_rcnn/
├── model/              # 模型定义
│   ├── resnet.py       # ResNet-50主干网络
│   ├── fpn.py          # FPN特征金字塔
│   └── plane_rcnn.py   # PlaneRCNN主模型
├── inference/          # 推理相关
│   ├── detector.py     # 平面检测器
│   └── post_processing.py # 结果后处理
├── projection/         # 投影相关
│   ├── projector.py    # 投影系统接口
│   ├── calibration.py  # 相机-投影机标定
│   └── visualizer.py   # 平面可视化与投影
└── utils/              # 工具类
    ├── camera.py       # 摄像头接口
    ├── motor.py        # 电机控制
    └── servo.py        # 舵机控制
```

## 3. 环境搭建

### 3.1 依赖安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install torch torchvision numpy opencv-python matplotlib
```

### 3.2 模型准备

PlaneRCNN模块使用现有的ResNet-50模型作为主干网络：

* 模型文件位置：`ResNet_project/models/config_a_epoch_1.pth`
* 模型定义：`ResNet_project/code/model_resnet.py`

## 4. 平面检测

### 4.1 图像平面检测

```python
from plane_rcnn.inference.detector import PlaneDetector
import cv2

# 1. 创建平面检测器
detector = PlaneDetector(
    model_path=None,  # 模型权重路径，None表示使用默认权重
    num_classes=2,     # 分类类别数
    device='cpu'       # 运行设备，可选'cpu'或'cuda'
)

# 2. 读取图像
image = cv2.imread('test_image.jpg')

# 3. 进行平面检测
results = detector.detect(image)

# 4. 结果解释
print(f"检测到 {len(results['planes'])} 个平面")
for i, plane in enumerate(results['planes']):
    print(f"平面 {i+1}:")
    print(f"  边界框: {plane['bbox']}")
    print(f"  置信度: {plane['confidence']:.2f}")
    print(f"  法向量: {plane['normal']}")
    print(f"  距离: {plane['distance']:.2f}")

# 5. 可视化结果
vis_image = detector.visualize_detections(image, results)

# 6. 保存结果
cv2.imwrite('detection_result.jpg', vis_image)
```

### 4.2 实时平面检测

```python
from plane_rcnn.utils.camera import Camera
from plane_rcnn.inference.detector import PlaneDetector
import cv2

# 1. 初始化摄像头
camera = Camera(config={
    'camera_id': 0,        # 摄像头ID
    'resolution': (640, 480),  # 分辨率
    'fps': 30              # 帧率
})
camera.open()

# 2. 初始化平面检测器
detector = PlaneDetector(device='cpu')

# 3. 实时检测循环
print("按ESC键退出实时检测")
while True:
    # 捕获图像
    frame = camera.capture_frame()
    
    if frame is not None:
        # 进行平面检测
        results = detector.detect(frame)
        
        # 可视化结果
        vis_frame = detector.visualize_detections(frame, results)
        
        # 显示结果
        cv2.imshow('Plane Detection', vis_frame)
    
    # 检测按键
    key = cv2.waitKey(1)
    if key == 27:  # ESC键
        break

# 4. 释放资源
camera.close()
cv2.destroyAllWindows()
```

## 5. 平面投影

### 5.1 投影系统初始化

```python
from plane_rcnn.projection.projector import Projector

# 1. 初始化投影系统
projector = Projector(config={
    'resolution': (1920, 1080),  # 投影分辨率
    'brightness': 50,             # 亮度(0-100)
    'contrast': 50                # 对比度(0-100)
})

# 2. 连接投影系统
projector.connect()

# 3. 获取投影系统状态
status = projector.get_status()
print(f"投影系统状态: {status}")
```

### 5.2 平面可视化投影

```python
from plane_rcnn.projection.visualizer import PlaneVisualizer

# 1. 初始化平面可视化器
visualizer = PlaneVisualizer()

# 2. 生成投影图像
proj_image = visualizer.generate_projection_image(
    results,              # 平面检测结果
    projection_matrix,    # 投影变换矩阵
    (1920, 1080)          # 投影分辨率
)

# 3. 投影图像
projector.project_image(proj_image)

# 4. 生成高亮投影
highlight_image = visualizer.generate_highlight_projection(
    results,              # 平面检测结果
    projection_matrix,    # 投影变换矩阵
    (1920, 1080)          # 投影分辨率
)

# 5. 投影高亮图像
projector.project_image(highlight_image)
```

### 5.3 相机-投影机标定

```python
from plane_rcnn.projection.calibration import CameraProjectorCalibrator

# 1. 初始化标定器
calibrator = CameraProjectorCalibrator(config={
    'checkerboard_size': (9, 6),  # 棋盘格尺寸
    'square_size': 25.0           # 棋盘格方格大小(mm)
})

# 2. 加载标定图像
camera_images = [cv2.imread(f'camera_image_{i}.jpg') for i in range(20)]
projector_images = [cv2.imread(f'projector_image_{i}.jpg') for i in range(20)]

# 3. 进行标定
calibrator.calibrate(camera_images, projector_images)

# 4. 保存标定结果
calibrator.save_calibration('calibration_result.npz')

# 5. 加载标定结果（后续使用）
calibrator.load_calibration('calibration_result.npz')
```

## 6. 结果后处理

```python
from plane_rcnn.inference.post_processing import PlanePostProcessor

# 1. 初始化后处理器
post_processor = PlanePostProcessor()

# 2. 后处理配置
config = {
    'refine_mask': True,              # 是否优化Mask
    'mask_refine_method': 'morphology',  # Mask优化方法
    'merge_overlapping': True,        # 是否合并重叠平面
    'iou_threshold': 0.5,             # 重叠阈值
    'filter_small': True,             # 是否过滤小面积平面
    'min_area': 100                   # 最小面积阈值(像素)
}

# 3. 进行后处理
processed_results = post_processor.post_process_results(results, config)

# 4. 结果过滤
filtered_planes = post_processor.filter_small_planes(
    processed_results['planes'], min_area=200
)

# 5. 计算平面面积
for plane in filtered_planes:
    area = post_processor.calculate_plane_area(plane['mask'])
    print(f"平面面积: {area} 像素")

# 6. 估计平面方向
for plane in filtered_planes:
    orientation = post_processor.estimate_plane_orientation(plane['normal'])
    print(f"平面方向: {orientation}")
```

## 7. 摄像头控制

```python
from plane_rcnn.utils.camera import Camera

# 1. 初始化摄像头
camera = Camera(config={
    'camera_id': 0,
    'resolution': (640, 480),
    'fps': 30
})

# 2. 打开摄像头
camera.open()

# 3. 显示摄像头状态
print(camera.get_status())

# 4. 捕获单帧图像
frame = camera.capture_frame()
camera.save_frame('captured_image.jpg', frame)

# 5. 捕获图像序列
frames = camera.capture_sequence(num_frames=10)
print(f"捕获了 {len(frames)} 帧图像")

# 6. 显示摄像头预览
# camera.show_preview()

# 7. 关闭摄像头
camera.close()
```

## 8. 硬件控制（预留接口）

### 8.1 电机控制

```python
from plane_rcnn.utils.motor import MotorController

# 1. 初始化电机控制器
motor_controller = MotorController(config={
    'motor_ids': [1, 2],     # 电机ID列表
    'max_speed': 100         # 最大速度
})

# 2. 连接电机控制器
motor_controller.connect()

# 3. 设置电机速度
motor_controller.set_motor_speed(1, 50)  # 电机1，速度50
motor_controller.set_motor_speed(2, -30) # 电机2，速度-30（反向）

# 4. 停止电机
motor_controller.stop_motor(1)
motor_controller.stop_all_motors()

# 5. 获取电机状态
status = motor_controller.get_motor_status(1)
print(f"电机1状态: {status}")

# 6. 断开电机控制器
motor_controller.disconnect()
```

### 8.2 舵机控制

```python
from plane_rcnn.utils.servo import ServoController

# 1. 初始化舵机控制器
servo_controller = ServoController(config={
    'servo_ids': [1, 2, 3, 4],  # 舵机ID列表
    'min_angle': 0,             # 最小角度
    'max_angle': 180,           # 最大角度
    'default_angle': 90         # 默认角度
})

# 2. 连接舵机控制器
servo_controller.connect()

# 3. 设置舵机角度
servo_controller.set_servo_angle(1, 45)  # 舵机1，角度45度
servo_controller.set_servo_angle(2, 135) # 舵机2，角度135度

# 4. 设置所有舵机角度
servo_controller.set_all_servos_angle(90)

# 5. 重置舵机到默认角度
servo_controller.reset_servos()

# 6. 获取舵机状态
status = servo_controller.get_servo_status(1)
print(f"舵机1状态: {status}")

# 7. 断开舵机控制器
servo_controller.disconnect()
```

## 9. 常见问题与解决方案

### 9.1 模型加载失败

**问题**：`加载模型权重失败: xxx`

**解决方案**：
1. 检查模型文件路径是否正确
2. 确保模型文件格式正确（PyTorch模型应为.pth或.pt格式）
3. 检查模型结构与代码是否匹配

### 9.2 检测效果不佳

**问题**：检测到的平面不准确或漏检

**解决方案**：
1. 确保图像质量良好，光线充足
2. 调整检测参数，如置信度阈值
3. 考虑对模型进行微调或训练
4. 优化后处理参数，如面积阈值

### 9.3 实时性能问题

**问题**：实时检测速度较慢

**解决方案**：
1. 使用GPU加速（设置device='cuda'）
2. 降低输入图像分辨率
3. 简化模型结构
4. 优化代码，减少不必要的计算

## 10. 示例代码

### 10.1 完整的平面检测与投影流程

```python
from plane_rcnn.utils.camera import Camera
from plane_rcnn.inference.detector import PlaneDetector
from plane_rcnn.projection.projector import Projector
from plane_rcnn.projection.visualizer import PlaneVisualizer
from plane_rcnn.projection.calibration import CameraProjectorCalibrator

# 1. 初始化组件
camera = Camera(config={'resolution': (640, 480)})
detector = PlaneDetector(device='cpu')
projector = Projector()
visualizer = PlaneVisualizer()
calibrator = CameraProjectorCalibrator()

# 2. 打开设备
camera.open()
projector.connect()

# 3. 加载标定结果
calibrator.load_calibration('calibration_result.npz')

# 4. 主循环
print("按ESC键退出")
while True:
    # 捕获图像
    frame = camera.capture_frame()
    
    if frame is not None:
        # 平面检测
        results = detector.detect(frame)
        
        # 可视化结果
        vis_frame = detector.visualize_detections(frame, results)
        cv2.imshow('Plane Detection', vis_frame)
        
        # 生成投影图像
        proj_image = visualizer.generate_projection_image(
            results, 
            calibrator.projection_matrix, 
            projector.projector_resolution
        )
        
        # 投影图像
        projector.project_image(proj_image)
    
    # 检测按键
    if cv2.waitKey(1) == 27:
        break

# 5. 释放资源
camera.close()
projector.disconnect()
cv2.destroyAllWindows()
```

## 11. 性能评估

### 11.1 模型评估

```bash
# 运行评估脚本
python plane_rcnn/experiments/evaluate.py --dataset path/to/dataset
```

### 11.2 评估指标

* **IoU**：交并比，评估Mask准确性
* **Precision/Recall**：评估检测准确性
* **平面参数误差**：评估法向量和距离的预测准确性
* **推理速度**：评估实时性能

## 12. 后续改进

1. **模型训练**：使用真实数据集训练模型
2. **实时性能优化**：使用TensorRT等加速推理
3. **多模态融合**：结合深度相机数据
4. **交互式平面检测**：允许用户交互选择平面
5. **动态场景适应**：适应动态变化的场景

## 13. 联系方式

如有问题或建议，请联系项目负责人。

---

**版本**：v1.0  
**更新时间**：2025-12-02  
**作者**：SphereMate团队