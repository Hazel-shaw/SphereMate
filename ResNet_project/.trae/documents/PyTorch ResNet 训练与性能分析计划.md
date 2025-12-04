# PyTorch ResNet 训练与性能分析计划

## 1. 项目结构设计

```
/Users/hazel/Documents/trae_projects/ResNet/
├── code/
│   ├── train.py           # 训练脚本
│   ├── model_resnet.py    # ResNet 模型定义
│   ├── dataset.py         # 数据集加载和预处理
│   ├── config.py          # 配置文件
│   └── utils.py           # 工具函数
├── models/                # 保存训练好的模型
├── data/                  # 数据集存储位置
├── report.md              # 性能分析报告
└── else.md                # 自己运行、训练步骤、其他细节
```

## 2. 数据集处理

* 使用 Kaggle 的 Fruit 数据集（fruits-360）

* 实现数据加载器，支持训练集、验证集和测试集划分

* 实现数据预处理和增强策略

## 3. ResNet 模型实现

### 3.1 基本模块

* 实现 BasicBlock（用于 ResNet-18/34）

* 实现 BottleneckBlock（用于 ResNet-50）

* 实现不同类型的 shortcut 连接

### 3.2 完整网络

* 实现可配置的 ResNet 类，支持不同深度

* 实现网络初始化和权重设置

## 4. 训练脚本实现

* 支持多种优化器（SGD、Adam、AdamW）

* 实现学习率调度器

* 实现训练循环和验证循环

* 实现模型保存和加载

* 记录训练日志和指标

## 5. 性能分析与实验

### 5.1 配置实验

* Config A: ResNet-18 + SGD + 基础配置

* Config B: ResNet-34 + Adam + 增大 batch size

* Config C: ResNet-50 + AdamW + 数据增强

* Config D: 自定义 ResNet + 改进激活函数

* Config E: 预训练 ResNet + 微调

### 5.2 评估指标

* 测试准确率

* 训练时间

* 过拟合情况

* 收敛速度

## 6. 报告编写

* 详细说明代码修改和实现细节

* 提供 ResNet 模型架构图

* 记录各配置的实验结果

* 分析不同配置对性能的影响

* 总结最佳实践和关键发现

## 7. 实施步骤

1. 创建项目目录结构
2. 下载并处理数据集
3. 实现 ResNet 模型
4. 实现训练脚本
5. 运行不同配置的实验
6. 收集实验结果
7. 编写最终报告
8. 另生成一个文档else.md（不包含在report中）给出我自己操作：训练推理的实施步骤，给出主要参考文献及项目中至关重要的逻辑+细节...用来辅助我答辩

