# PlaneRCNN答辩文稿
- torch ：深度学习框架
- torchvision ：计算机视觉工具库
- numpy ：数值计算库
- opencv-python ：图像处理库
- matplotlib ：可视化库
## 1. 论文核心思想与结构

### 1.1 论文概述

* **标题**：PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image

* **作者**：Chen Liu, Kihwan Kim, Jinwei Gu, Yasutaka Furukawa, Jan Kautz

* **机构**：NVIDIA, Washington University in St. Louis

* **发表时间**：2019年

* **核心贡献**：
  - 提出了一种从单图像中同时检测平面实例并预测其3D参数的深度学习方法
  - 扩展了Mask R-CNN架构，添加了平面参数预测分支
  - 引入了跨视角一致性损失，提高了平面检测的准确性
  - 实现了从2D图像到3D平面的直接映射

### 1.2 网络架构

PlaneRCNN基于Mask R-CNN架构，主要包含以下部分：

1. **ResNet主干网络**：用于提取图像特征
2. **特征金字塔网络(FPN)**：生成多尺度特征图
3. **区域建议网络(RPN)**：生成区域建议
4. **检测头**：包含分类、边界框回归和Mask生成
5. **平面参数预测分支**：预测平面法向量和距离

### 1.3 平面表示

平面使用**法向量+距离**的参数化表示：

* 法向量(nx, ny, nz)：描述平面的朝向
* 距离d：平面到相机的距离

平面方程：nx*x + ny*y + nz*z = d

### 1.4 损失函数

PlaneRCNN使用多任务损失函数：

* **检测损失**：分类损失(CrossEntropyLoss) + 边界框回归损失(SmoothL1Loss)
* **Mask损失**：二进制交叉熵损失(BCEWithLogitsLoss)
* **平面参数损失**：法向量损失(余弦相似度损失) + 距离损失(L1Loss)
* **跨视角一致性损失**：增强不同视角下平面检测的一致性

## 2. 知识点总结

### 2.1 深度学习基础知识

#### 2.1.1 ResNet

* **残差连接**：解决深层网络训练中的梯度消失问题
* **网络结构**：由多个残差块组成，包含BasicBlock和BottleneckBlock
* **特征提取**：通过卷积层逐步提取图像特征，输出多尺度特征图

#### 2.1.2 Mask R-CNN

* **架构**：Faster R-CNN + Mask分支
* **实例分割**：同时完成目标检测和语义分割
* **RoIAlign**：精确提取感兴趣区域特征，解决RoIPool的量化误差

#### 2.1.3 特征金字塔网络(FPN)

* **自下而上路径**：ResNet主干网络提取特征
* **自上而下路径**：高层特征通过上采样与低层特征融合
* **横向连接**：将不同尺度的特征图转换为相同通道数
* **多尺度特征**：生成不同尺度的特征图，用于检测不同大小的目标

### 2.2 平面检测与重建

#### 2.2.1 单图像3D重建

* **挑战**：从2D图像中恢复3D信息是一个病态问题
* **解决方案**：利用深度学习学习从2D图像到3D参数的映射
* **应用**：增强现实、机器人导航、室内场景理解

#### 2.2.2 平面检测

* **实例平面检测**：检测图像中的每个平面实例
* **平面分割**：生成每个平面的像素级分割掩码
* **平面参数预测**：预测每个平面的3D参数

### 2.3 投影几何

#### 2.3.1 相机模型

* **针孔相机模型**：将3D空间点投影到2D图像平面
* **内参矩阵**：描述相机的固有属性
* **外参矩阵**：描述相机在世界坐标系中的位置和姿态

#### 2.3.2 相机-投影机标定

* **标定目的**：建立相机图像与投影图像之间的映射关系
* **标定方法**：棋盘格标定法
* **标定过程**：
  1. 拍摄包含棋盘格的图像
  2. 检测棋盘格角点
  3. 计算相机内参和外参
  4. 计算投影机内参和外参
  5. 计算相机到投影机的变换矩阵

## 3. 项目实现

### 3.1 现有ResNet-50模型集成

* **模型位置**：`ResNet_project/code/model_resnet.py`
* **集成方式**：通过`get_resnet50_backbone()`函数导入模型，去除分类层，保留特征提取部分
* **输出特征**：输入224x224 → 输出7x7x2048

### 3.2 PlaneRCNN实现

* **主干网络**：ResNet-50
* **特征金字塔**：FPN，输出5个尺度的特征图
* **检测头**：简化版RPN + 分类回归头 + Mask头 + 平面参数头
* **平面参数**：法向量(3维) + 距离(1维)

### 3.3 推理流程

1. **图像预处理**：resize → 归一化 → 转换为Tensor
2. **特征提取**：ResNet-50 + FPN
3. **区域建议**：RPN生成候选区域
4. **检测与分割**：分类 + 边界框回归 + Mask生成
5. **平面参数预测**：法向量和距离预测
6. **后处理**：Mask优化 → 平面面积计算 → 方向估计 → 结果筛选

### 3.4 投影功能

* **平面可视化**：高亮区域 + 边界框 + 文字标注
* **投影映射**：相机图像到投影图像的变换
* **提示信息**：检测到的平面类型(地面、墙面、桌面等)

## 4. 操作指南

### 4.1 环境搭建

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活环境
source venv/bin/activate

# 3. 安装依赖
pip install torch torchvision numpy opencv-python matplotlib

# 4. 进入项目目录
cd SphereMate
```

### 4.2 模型推理

#### 4.2.1 图像平面检测

```python
from plane_rcnn.inference.detector import PlaneDetector
import cv2

# 1. 创建平面检测器
detector = PlaneDetector(device='cpu')

# 2. 读取图像
image = cv2.imread('test_image.jpg')

# 3. 进行平面检测
results = detector.detect(image)

# 4. 可视化检测结果
vis_image = detector.visualize_detections(image, results)

# 5. 保存结果
cv2.imwrite('result.jpg', vis_image)
```

#### 4.2.2 实时平面检测

```python
from plane_rcnn.utils.camera import Camera
from plane_rcnn.inference.detector import PlaneDetector
import cv2

# 1. 创建摄像头
camera = Camera(config={'resolution': (640, 480)})
camera.open()

# 2. 创建平面检测器
detector = PlaneDetector(device='cpu')

# 3. 实时检测
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
    
    # 按ESC键退出
    if cv2.waitKey(1) == 27:
        break

# 4. 释放资源
camera.close()
cv2.destroyAllWindows()
```

### 4.3 投影功能

```python
from plane_rcnn.projection.projector import Projector
from plane_rcnn.projection.visualizer import PlaneVisualizer

# 1. 创建投影机
projector = Projector()
projector.connect()

# 2. 创建平面可视化器
visualizer = PlaneVisualizer()

# 3. 生成投影图像
proj_image = visualizer.generate_projection_image(
    results, projection_matrix, (1920, 1080)
)

# 4. 投影图像
projector.project_image(proj_image)

# 5. 断开投影机连接
projector.disconnect()
```

### 4.4 模型评估

```python
# 运行评估脚本
python plane_rcnn/experiments/evaluate.py --dataset path/to/dataset
```

## 5. 实验设计与结果

### 5.1 实验数据

* **测试图像**：使用水果图像数据集进行测试
* **数据集位置**：`ResNet_project/data/fruits-360_multi/test-multiple_fruits/`

### 5.2 评估指标

* **IoU**：交并比，评估Mask准确性
* **Precision/Recall**：评估检测准确性
* **平面参数误差**：评估法向量和距离的预测准确性

### 5.3 预期结果

* 准确检测图像中的平面实例
* 生成高质量的平面Mask
* 精确预测平面参数
* 实时性能满足应用需求

## 6. 问题与反思

### 6.1 技术挑战

* **单图像3D重建**：从2D图像恢复3D信息是一个病态问题
* **平面参数预测**：法向量和距离的准确预测难度较大
* **实时性能**：深度学习模型的推理速度需要优化
* **相机-投影机标定**：精确标定需要专业设备和复杂流程

### 6.2 解决方案

* **数据增强**：使用多视角数据增强模型泛化能力
* **损失函数优化**：设计更适合平面参数预测的损失函数
* **模型优化**：使用轻量级网络或模型压缩技术
* **简化标定**：开发简化的相机-投影机标定方法

## 7. 后续改进方向

1. **模型训练**：使用真实数据集训练模型，提高检测准确性
2. **实时性能优化**：使用TensorRT等加速推理
3. **多模态融合**：结合深度相机数据提高3D重建准确性
4. **交互式平面检测**：允许用户交互选择平面
5. **动态场景适应**：适应动态变化的场景
6. **投影系统优化**：提高投影精度和亮度

## 8. 项目总结

本项目成功实现了基于PlaneRCNN的平面检测与投影系统，主要完成了以下工作：

1. 集成了现有ResNet-50模型作为主干网络
2. 实现了FPN特征金字塔和Mask R-CNN检测头
3. 添加了平面参数预测分支
4. 实现了完整的推理流程
5. 开发了平面投影功能
6. 预留了硬件控制接口

项目为SphereMate平台提供了强大的平面检测和投影能力，可应用于增强现实、机器人导航等领域。