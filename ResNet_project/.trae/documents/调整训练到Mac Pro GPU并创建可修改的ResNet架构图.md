# 调整训练到Mac Pro GPU并创建可修改的ResNet架构图

## 1. 调整训练到Mac Pro GPU

### 1.1 修改设备获取逻辑

* 修改 `utils.py` 中的 `get_device()` 函数

* 添加对Apple Silicon MPS后端的支持

* 优先使用MPS（Mac GPU），其次是CUDA，最后是CPU

### 1.2 代码修改

```python
def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    返回:
        device: torch.device对象，表示可用的设备
    """
    # 检查是否支持MPS（Apple Silicon GPU）
    if torch.backends.mps.is_available():
        return torch.device('mps')
    # 检查是否支持CUDA（NVIDIA GPU）
    elif torch.cuda.is_available():
        return torch.device('cuda')
    # 否则使用CPU
    else:
        return torch.device('cpu')
```

## 2. 创建可修改的ResNet架构图

### 2.1 技术选择

* 使用HTML + CSS + JavaScript创建交互式架构图

* 使用SVG绘制架构图元素，便于修改和交互

* 添加简单的编辑功能，允许调整节点大小、颜色和文本

### 2.2 架构图内容

* 完整的ResNet架构，包括：

  * 输入层

  * 卷积层、批归一化层、ReLU激活层

  * 最大池化层

  * 四个阶段的残差块

  * 自适应平均池化层

  * 全连接层

* 清晰标示各层输入输出维度

* 区分不同类型的层（卷积、池化、残差块等）

* 显示stage划分

* 标示shortcut类型（identity / projection）

### 2.3 文件结构

* 创建 `resnet_architecture.html` 文件

* 包含完整的HTML、CSS和JavaScript代码

* 支持直接在浏览器中打开和修改

### 2.4 交互功能

* 点击节点可编辑文本

* 拖拽调整节点位置

* 支持缩放和平移

* 可切换显示/隐藏维度信息

* 可导出为图片

## 3. 更新报告

* 在 `report.md` 中添加对HTML架构图的引用

* 说明如何使用和修改该架构图

* 提供示例截图

## 4. 测试和验证

* 在Mac Pro 14上测试训练脚本，确保使用MPS后端

* 测试HTML架构图的交互功能

* 验证架构图与报告中的描述一致

## 5. 交付物

* 修改后的 `utils.py` 文件

* `resnet_architecture.html` 文件

* 更新后的 `report.md` 文件

这个计划将确保训练脚本能够利用Mac Pro 14的GPU加速，并提供一个直观、可修改的ResNet架构图，便于理解和调整。

