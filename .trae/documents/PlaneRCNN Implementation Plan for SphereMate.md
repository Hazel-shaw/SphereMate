# SphereMate项目PlaneRCNN实现计划

## 1. 项目概述

本项目旨在实现PlaneRCNN算法用于单图像3D平面检测与重建，并将其集成到SphereMate平台中，具备平面投影功能。项目将使用现有的ResNet-50模型作为主干网络。

## 2. 现有ResNet-50模型分析

### 2.1 模型结构

* 位置：`ResNet_project/code/model_resnet.py`

* 实现了完整的ResNet架构，包括：

  * BottleneckBlock（用于ResNet-50/101/152）

  * 完整的ResNet前向传播（conv1 + bn1 + relu + maxpool + layer1-4 + avgpool + fc）

  * 模型构建函数：`resnet50()`

### 2.2 集成优势

* 现有模型结构完整，易于修改

* 支持ResNet-50，符合项目要求

* 代码清晰，便于维护和扩展

* 已在PyTorch环境中运行，兼容性好

## 3. PlaneRCNN实现计划

### 3.1 核心架构

```
输入图像 → ResNet-50主干网络 → FPN特征金字塔 → Mask R-CNN检测头 → 平面参数预测分支
```

### 3.2 实现步骤

#### 3.2.1 模型集成

* **创建平面检测模型目录**：`mkdir -p plane_rcnn/model`

* **导入现有ResNet-50**：

  ```python
  # plane_rcnn/model/resnet.py
  import sys
  sys.path.append('../ResNet_project/code')
  from model_resnet import resnet50
  
  def get_resnet50_backbone(pretrained=True):
      """获取ResNet-50主干网络，去除分类层"""
      model = resnet50()
      if pretrained:
          # 加载预训练权重
          model.load_state_dict(torch.load('../ResNet_project/models/config_a_epoch_1.pth'))
      # 去除分类层，保留特征提取部分
      backbone = nn.Sequential(
          model.conv1,
          model.bn1,
          model.relu,
          model.maxpool,
          model.layer1,
          model.layer2,
          model.layer3,
          model.layer4
      )
      return backbone
  ```

* **实现FPN特征金字塔**：

  ```python
  # plane_rcnn/model/fpn.py
  class FPN(nn.Module):
      def __init__(self, in_channels_list, out_channels):
          super(FPN, self).__init__()
          # 实现FPN上采样和侧向连接
          # ...
      def forward(self, x):
          # 前向传播，生成多尺度特征
          # ...
  ```

* **构建PlaneRCNN主模型**：

  ```python
  # plane_rcnn/model/plane_rcnn.py
  class PlaneRCNN(nn.Module):
      def __init__(self, backbone, fpn):
          super(PlaneRCNN, self).__init__()
          self.backbone = backbone
          self.fpn = fpn
          # 实现RPN、检测头、Mask分支、平面参数分支
          # ...
      def forward(self, x):
          # 前向传播
          # ...
  ```

#### 3.2.2 推理流程

* **图像预处理**：

  ```python
  # plane_rcnn/inference/detector.py
  def preprocess(image):
      # 缩放、归一化等预处理
      # ...
  ```

* **模型推理**：

  ```python
  def detect_planes(image, model):
      # 预处理
      input_tensor = preprocess(image)
      # 模型前向传播
      outputs = model(input_tensor)
      # 后处理
      planes = post_process(outputs)
      return planes
  ```

* **后处理**：

  ```python
  # plane_rcnn/inference/post_processing.py
  def post_process(outputs):
      # 解码预测结果
      # 生成平面mask
      # 计算平面参数
      # ...
  ```

#### 3.2.3 投影功能

* **平面可视化**：

  ```python
  # plane_rcnn/projection/visualizer.py
  def visualize_planes(image, planes):
      # 绘制平面高亮区域
      # 绘制边界框
      # 添加文字标注
      # ...
  ```

* **相机-投影机标定**：

  ```python
  # plane_rcnn/projection/calibration.py
  def calibrate_camera_projector():
      # 使用棋盘格进行标定
      # 计算内参和外参
      # 生成映射矩阵
      # ...
  ```

## 4. 答辩文稿准备

### 4.1 论文核心思想与结构

#### 4.1.1 论文概述

* **标题**：PlaneRCNN: 3D Plane Detection and Reconstruction from a Single Image

* **作者**：Chen Liu, Kihwan Kim, Jinwei Gu, Yasutaka Furukawa, Jan Kautz

* **机构**：NVIDIA, Washington University in St. Louis

* **核心贡献**：

  * 提出了一种从单图像中同时检测平面实例并预测其3D参数的深度学习方法

  * 扩展了Mask R-CNN架构，添加了平面参数预测分支

  * 引入了跨视角一致性损失，提高了平面检测的准确性

#### 4.1.2 网络架构

* **主干网络**：ResNet + FPN

* **检测头**：

  * RPN（区域建议网络）

  * 分类和边界框回归

  * Mask生成

  * 平面参数预测（法向量 + 距离）

#### 4.1.3 损失函数

* **检测损失**：分类损失 + 边界框回归损失

* **Mask损失**：二进制交叉熵损失

* **平面参数损失**：法向量损失 + 距离损失

* **跨视角一致性损失**：增强不同视角下平面检测的一致性

### 4.2 知识点总结

#### 4.2.1 深度学习基础知识

* **ResNet**：残差连接解决梯度消失问题

* **Mask R-CNN**：实例分割的经典算法

* **FPN**：特征金字塔网络，融合多尺度特征

#### 4.2.2 平面检测与重建

* **平面表示**：法向量 + 距离

* **单图像3D重建**：从2D图像中恢复3D信息

* **投影几何**：相机模型、单应性映射

#### 4.2.3 相机-投影机标定

* **标定原理**：透视变换、相机内参和外参

* **标定方法**：棋盘格标定法

* **映射计算**：从图像平面到投影平面的变换

### 4.3 操作指南

#### 4.3.1 环境搭建

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
source venv/bin/activate

# 安装依赖
pip install torch torchvision numpy opencv-python matplotlib

# 克隆项目
git clone https://github.com/hazel/SphereMate.git
cd SphereMate
```

#### 4.3.2 模型推理

```python
# 运行平面检测
python main.py --mode detect --image path/to/image.jpg

# 运行实时检测
python main.py --mode realtime

# 运行投影功能
python main.py --mode project
```

#### 4.3.3 评估与可视化

```python
# 评估平面检测精度
python plane_rcnn/experiments/evaluate.py --dataset path/to/dataset

# 可视化检测结果
python plane_rcnn/projection/visualizer.py --image path/to/image.jpg --output path/to/output.jpg
```

### 4.4 预期结果与分析

* **平面检测效果**：

  * 准确检测图像中的平面实例

  * 生成高质量的平面mask

  * 精确预测平面参数

* **投影效果**：

  * 平面高亮区域与物理平面准确对齐

  * 边界框清晰可见

  * 文字标注位置准确

* **评估指标**：

  * IoU（交并比）

  * precision/recall

  * 平面参数误差

## 5. 文件结构

```
SphereMate/
├── plane_rcnn/
│   ├── model/
│   │   ├── resnet.py          # 现有ResNet-50集成
│   │   ├── fpn.py             # FPN特征金字塔
│   │   ├── mask_rcnn.py       # Mask R-CNN检测头
│   │   └── plane_rcnn.py      # PlaneRCNN主架构
│   ├── inference/
│   │   ├── detector.py        # 平面检测推理
│   │   └── post_processing.py # 后处理
│   ├── projection/
│   │   ├── projector.py       # 投影系统接口
│   │   ├── calibration.py     # 相机-投影机标定
│   │   └── visualizer.py      # 平面可视化与投影
│   ├── utils/
│   │   ├── camera.py          # 摄像头接口
│   │   ├── motor.py           # 电机控制预留代码
│   │   └── servo.py           # 舵机控制预留代码
│   └── experiments/
│       ├── evaluate.py        # 模型评估
│       └── results/           # 实验结果
├── docs/
│   ├── report.md              # 期中项目报告
│   └── defense_slides.md      # 答辩幻灯片
├── main.py                    # 主应用入口
└── ResNet_project/            # 现有ResNet项目
    └── code/
        └── model_resnet.py    # 现有ResNet实现
```

## 6. 预期成果

* **核心模型**：基于现有ResNet-50的PlaneRCNN实现

* **集成系统**：与SphereMate平台集成的平面检测与投影系统

* **答辩文稿**：详细的论文分析、知识点总结和操作指南

* **实验报告**：包含平面检测效果评估和可视化结果

## 7. 后续改进方向

* 模型训练与微调

* 投影系统优化

* 多平面交互功能

* 实时性能优化

* 更复杂场景下的平面检测

