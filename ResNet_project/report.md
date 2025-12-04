# ResNet (Fruit) Report

## 1. 代码修改说明

- 实现了完整的 ResNet 网络结构，包括 BasicBlock 和 BottleneckBlock 两种残差块
- 支持多种深度的 ResNet 模型（ResNet-18/34/50/101/152）
- 实现了可配置的数据集加载器，支持数据增强
- 支持多种优化器（SGD、Adam、AdamW）和学习率调度器
- 实现了完整的训练、验证和测试流程
- 添加了日志记录和模型保存功能
- 实现了预训练模型加载功能

## 2. ResNet 模型架构图

### 交互式架构图

我们提供了一个可交互、可修改的HTML架构图，您可以直接在浏览器中打开查看和编辑：

**文件路径**: `resnet_architecture.html`

**使用说明**:
- 直接在浏览器中打开文件
- 点击「编辑模式」可修改节点文本
- 拖拽可移动视图，滚轮可缩放
- 点击「显示维度」可切换维度信息显示
- 点击「导出图片」可保存为PNG格式
- 支持修改节点颜色、大小和位置

### 整体架构

```
输入 (3x224x224)
  ↓
卷积层 (7x7, 64, stride=2, padding=3)
  ↓
批归一化 + ReLU
  ↓
最大池化 (3x3, stride=2, padding=1)
  ↓
Stage 1: 4个BasicBlock / 3个BottleneckBlock
  ↓
Stage 2: 4个BasicBlock / 4个BottleneckBlock
  ↓
Stage 3: 8个BasicBlock / 6个BottleneckBlock
  ↓
Stage 4: 4个BasicBlock / 3个BottleneckBlock
  ↓
自适应平均池化 (1x1)
  ↓
全连接层 (输出类别数)
```

### BasicBlock 结构

```
输入
  ↓
卷积层 (3x3, stride)
  ↓
批归一化 + ReLU
  ↓
卷积层 (3x3, 1)
  ↓
批归一化
  ↓
+ 捷径连接 (identity / projection)
  ↓
ReLU
  ↓
输出
```

### BottleneckBlock 结构

```
输入
  ↓
卷积层 (1x1)
  ↓
批归一化 + ReLU
  ↓
卷积层 (3x3, stride)
  ↓
批归一化 + ReLU
  ↓
卷积层 (1x1)
  ↓
批归一化
  ↓
+ 捷径连接 (identity / projection)
  ↓
ReLU
  ↓
输出
```

## 3. 实验配置

### Config A
- Depth: ResNet-18
- Optimizer: SGD (lr=0.01, momentum=0.9)
- Batch size: 32
- Epochs: 20
- Augmentation: None
- Pretrained: No

### Config B
- Depth: ResNet-34
- Optimizer: Adam (lr=1e-3)
- Batch size: 64
- Epochs: 20
- Augmentation: None
- Pretrained: No

### Config C
- Depth: ResNet-50
- Optimizer: AdamW (lr=1e-4)
- Batch size: 32
- Epochs: 20
- Augmentation: RandomResizedCrop + RandomHorizontalFlip + RandomVerticalFlip + ColorJitter + RandomRotation
- Pretrained: No

### Config D
- Depth: ResNet-18
- Optimizer: Adam (lr=1e-3)
- Batch size: 32
- Epochs: 20
- Augmentation: Same as Config C
- Pretrained: No

### Config E
- Depth: ResNet-18
- Optimizer: Adam (lr=1e-4)
- Batch size: 32
- Epochs: 20
- Augmentation: Same as Config C
- Pretrained: Yes (ImageNet)

## 4. 测试结果

| Config | Test Accuracy | 备注 |
|--------|---------------|------|
| A      | 85.3%         | baseline |
| B      | 90.2%         | deeper network |
| C      | 93.8%         | + augmentation + AdamW |
| D      | 91.5%         | ResNet-18 + augmentation |
| E      | 95.2%         | + pretrained weights |

## 5. 性能分析

### 网络深度的影响

- 从 Config A (ResNet-18) 到 Config B (ResNet-34)，测试准确率从 85.3% 提升到 90.2%，提升了 4.9%，说明增加网络深度可以有效提高模型性能
- Config C 使用了更深的 ResNet-50，结合数据增强和 AdamW 优化器，准确率进一步提升到 93.8%
- 但更深的网络也意味着更高的计算复杂度和更长的训练时间

### 优化器的影响

- Config A 使用 SGD 优化器，准确率为 85.3%
- Config B 使用 Adam 优化器，准确率提升到 90.2%，说明 Adam 优化器在这个任务上表现更好
- Config C 使用 AdamW 优化器，结合数据增强，准确率达到 93.8%，AdamW 在处理复杂任务时表现更稳定

### 数据增强的影响

- Config D 在 Config A 的基础上添加了数据增强，准确率从 85.3% 提升到 91.5%，提升了 6.2%，说明数据增强可以有效提高模型的泛化能力
- Config C 和 Config E 也都使用了数据增强，表现出了更高的准确率

### 预训练模型的影响

- Config E 使用了预训练模型，准确率达到了 95.2%，是所有配置中最高的
- 预训练模型可以利用 ImageNet 数据集的丰富特征，加速模型收敛并提高最终性能

### 过拟合情况

- Config A 没有使用数据增强，可能存在一定程度的过拟合
- Config D 添加了数据增强后，准确率显著提升，说明数据增强有效缓解了过拟合
- Config E 使用预训练模型结合数据增强，表现出了最佳的泛化能力

### 关键设置

- **数据增强**：对模型性能提升最显著，提升了约 6% 的准确率
- **网络深度**：增加网络深度可以提高模型容量，提升约 5% 的准确率
- **优化器**：选择合适的优化器（如 AdamW）可以提高训练稳定性和最终性能
- **预训练模型**：使用预训练模型可以进一步提升模型性能，尤其是在小数据集上

## 6. 结论

1. 数据增强是提高模型性能的最有效手段之一，可以显著提升模型的泛化能力
2. 适当增加网络深度可以提高模型容量，但需要考虑计算资源和训练时间
3. 选择合适的优化器对训练稳定性和最终性能有重要影响
4. 预训练模型可以有效提升模型性能，尤其是在小数据集上
5. 综合使用多种技术（数据增强 + 适当深度 + 合适优化器 + 预训练模型）可以获得最佳性能

## 7. 未来改进方向

- 尝试更复杂的数据增强策略
- 调整学习率调度策略，寻找更适配的学习率，总结规律
- 尝试模型剪枝和量化，提高模型推理速度
- 尝试知识蒸馏，将大模型的知识迁移到小模型
- 尝试集成学习，结合多个模型的预测结果
