# 导入os模块，用于文件路径操作
import os

# 数据集配置
data_config = {
    # 数据集根目录路径
    'data_dir': '../data/fruits-360_100x100/fruits-360',
    # 数据加载时使用的工作进程数，用于并行加载数据
    'num_workers': 4,
}

# 模型配置
model_config = {
    # 模型名称，可选值：resnet18, resnet34, resnet50, resnet101, resnet152
    'model_name': 'resnet18',
    # 是否使用预训练模型
    'pretrained': False,
    # 是否将残差分支的最后一个BN层初始化为0，有助于模型收敛
    'zero_init_residual': False,
}

# 训练配置
train_config = {
    # 批次大小，每个批次包含的样本数量
    'batch_size': 32,
    # 训练的epoch数量，每个epoch表示遍历整个训练集一次
    'epochs': 20,
    # 初始学习率
    'learning_rate': 0.001,
    # 权重衰减（L2正则化），用于防止过拟合
    'weight_decay': 0.0001,
    # 优化器类型，可选值：sgd, adam, adamw
    'optimizer': 'adam',
    # 学习率调度器类型，可选值：step, cosine, plateau
    'scheduler': 'step',
    # 学习率衰减步长（仅用于step调度器），每7个epoch衰减一次
    'step_size': 7,
    # 学习率衰减因子，每次衰减时乘以0.1
    'gamma': 0.1,
    # 是否使用数据增强
    'augment': False,
}

# 实验配置
experiments = {
    # 实验名称：config_a - 基础配置，快速测试
    'config_a': {
        'model_name': 'resnet18',        # 使用ResNet-18模型
        'optimizer': 'sgd',              # 使用SGD优化器
        'batch_size': 32,                # 批次大小32
        'epochs': 1,                     # 训练1个epoch（快速测试）
        'learning_rate': 0.01,           # 学习率0.01
        'augment': False,                # 不使用数据增强
        'pretrained': False,             # 不使用预训练模型
    },
    # 实验名称：config_b - ResNet34 + Adam优化器
    'config_b': {
        'model_name': 'resnet34',        # 使用ResNet-34模型
        'optimizer': 'adam',             # 使用Adam优化器
        'batch_size': 64,                # 批次大小64
        'epochs': 1,                     # 训练1个epoch（快速测试）
        'learning_rate': 0.001,          # 学习率0.001
        'augment': True,                 # 使用数据增强
        'pretrained': False,             # 不使用预训练模型
    },
    # 实验名称：config_c - ResNet50 + 预训练模型
    'config_c': {
        'model_name': 'resnet50',        # 使用ResNet-50模型
        'optimizer': 'adamw',            # 使用AdamW优化器
        'batch_size': 32,                # 批次大小32
        'epochs': 1,                     # 训练1个epoch（快速测试）
        'learning_rate': 0.0001,         # 学习率0.0001
        'augment': True,                 # 使用数据增强
        'pretrained': True,              # 使用预训练模型
    },
    # 实验名称：config_d - ResNet18 + 数据增强
    'config_d': {
        'model_name': 'resnet18',        # 使用ResNet-18模型
        'optimizer': 'sgd',              # 使用SGD优化器
        'batch_size': 64,                # 批次大小64
        'epochs': 1,                     # 训练1个epoch（快速测试）
        'learning_rate': 0.005,          # 学习率0.005
        'augment': True,                 # 使用数据增强
        'pretrained': False,             # 不使用预训练模型
    },
}

# 输出配置
output_config = {
    # 模型保存目录
    'model_dir': '../models',
    # 日志保存目录
    'log_dir': '../logs',
    # 模型保存频率，每5个epoch保存一次模型
    'save_freq': 5,
}

# 创建输出目录
for dir_path in [output_config['model_dir'], output_config['log_dir']]:
    # 如果目录不存在，创建目录
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 为了兼容旧代码，保留大写变量名
DATA_CONFIG = data_config
MODEL_CONFIG = model_config
TRAIN_CONFIG = train_config
EXPERIMENTS = experiments
OUTPUT_CONFIG = output_config