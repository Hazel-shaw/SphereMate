# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch核心库
import logging  # 用于日志记录
from torch import optim  # 用于优化器
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau  # 用于学习率调度器
# 从model_resnet.py导入ResNet模型
from model_resnet import resnet18, resnet34, resnet50, resnet101, resnet152

# 定义设置日志记录的函数
def setup_logging(log_dir, experiment_name):
    """
    设置日志记录，用于记录训练过程中的信息
    
    参数:
        log_dir: 字符串，表示日志文件的保存目录
        experiment_name: 字符串，表示实验名称，用于命名日志文件
        
    返回:
        logger: 日志记录器对象，用于记录日志
    """
    # 构建日志文件路径
    log_file = os.path.join(log_dir, f'{experiment_name}.log')
    
    # 创建日志目录，如果不存在的话
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志记录器，使用实验名称作为名称
    logger = logging.getLogger(experiment_name)
    # 设置日志级别为INFO，只记录INFO及以上级别的日志
    logger.setLevel(logging.INFO)
    
    # 清除之前的处理器，避免重复记录
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建文件处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式：时间 - 日志名称 - 日志级别 - 日志内容
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 将格式应用到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 定义获取模型的函数
def get_model(model_name, num_classes, pretrained=False, zero_init_residual=False):
    """
    获取ResNet模型
    
    参数:
        model_name: 字符串，表示模型名称，可选值：resnet18, resnet34, resnet50, resnet101, resnet152
        num_classes: 整数，表示分类的类别数量
        pretrained: 布尔值，表示是否使用预训练模型
        zero_init_residual: 布尔值，表示是否将残差分支的最后一个BN层初始化为0
        
    返回:
        model: ResNet模型对象
    """
    # 创建模型字典，映射模型名称到对应的模型函数
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
    }
    
    # 检查模型名称是否有效
    if model_name not in model_dict:
        raise ValueError(f"无效的模型名称: {model_name}")
    
    # 创建模型实例
    model = model_dict[model_name](num_classes=num_classes, zero_init_residual=zero_init_residual)
    
    # 如果需要使用预训练模型
    if pretrained:
        # 从torchvision.models导入预训练模型
        from torchvision.models import (
            resnet18 as tv_resnet18,
            resnet34 as tv_resnet34,
            resnet50 as tv_resnet50,
            resnet101 as tv_resnet101,
            resnet152 as tv_resnet152
        )
        
        # 创建预训练模型字典
        tv_model_dict = {
            'resnet18': tv_resnet18,
            'resnet34': tv_resnet34,
            'resnet50': tv_resnet50,
            'resnet101': tv_resnet101,
            'resnet152': tv_resnet152,
        }
        
        # 加载预训练模型
        tv_model = tv_model_dict[model_name](pretrained=True)
        # 获取预训练模型的状态字典（权重）
        pretrained_state_dict = tv_model.state_dict()
        # 获取当前模型的状态字典
        model_state_dict = model.state_dict()
        
        # 只加载匹配的权重，跳过不匹配的层（如最后一层全连接层）
        for key in pretrained_state_dict:
            # 检查键是否存在且形状匹配
            if key in model_state_dict and pretrained_state_dict[key].shape == model_state_dict[key].shape:
                model_state_dict[key] = pretrained_state_dict[key]
        
        # 将预训练权重加载到当前模型
        model.load_state_dict(model_state_dict)
    
    return model

# 定义获取优化器的函数
def get_optimizer(model, optimizer_name, learning_rate, weight_decay):
    """
    获取优化器
    
    参数:
        model: 模型对象
        optimizer_name: 字符串，表示优化器名称，可选值：sgd, adam, adamw
        learning_rate: 浮点数，表示学习率
        weight_decay: 浮点数，表示权重衰减（L2正则化）
        
    返回:
        optimizer: 优化器对象
    """
    # 根据优化器名称选择优化器
    if optimizer_name == 'sgd':
        # SGD优化器，带有动量
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        # Adam优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        # AdamW优化器，Adam + L2正则化
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"无效的优化器名称: {optimizer_name}")
    
    return optimizer

# 定义获取学习率调度器的函数
def get_scheduler(optimizer, scheduler_name, step_size=7, gamma=0.1):
    """
    获取学习率调度器
    
    参数:
        optimizer: 优化器对象
        scheduler_name: 字符串，表示调度器名称，可选值：step, cosine, plateau
        step_size: 整数，表示学习率衰减的步长（仅用于step调度器）
        gamma: 浮点数，表示学习率衰减因子
        
    返回:
        scheduler: 学习率调度器对象
    """
    # 根据调度器名称选择调度器
    if scheduler_name == 'step':
        # StepLR：每step_size个epoch，学习率乘以gamma
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'cosine':
        # CosineAnnealingLR：余弦退火调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=200)
    elif scheduler_name == 'plateau':
        # ReduceLROnPlateau：当指标停止改善时，降低学习率
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=gamma, patience=5)
    else:
        raise ValueError(f"无效的调度器名称: {scheduler_name}")
    
    return scheduler

# 定义保存模型的函数
def save_model(model, model_dir, experiment_name, epoch):
    """
    保存模型权重
    
    参数:
        model: 模型对象
        model_dir: 字符串，表示模型保存目录
        experiment_name: 字符串，表示实验名称
        epoch: 整数，表示当前epoch数
    """
    # 创建模型目录，如果不存在的话
    os.makedirs(model_dir, exist_ok=True)
    # 构建模型保存路径
    model_path = os.path.join(model_dir, f'{experiment_name}_epoch_{epoch}.pth')
    # 保存模型权重
    torch.save(model.state_dict(), model_path)

# 定义加载模型的函数
def load_model(model, model_path):
    """
    加载模型权重
    
    参数:
        model: 模型对象
        model_path: 字符串，表示模型权重文件路径
        
    返回:
        model: 加载权重后的模型对象
    """
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    return model

# 定义计算准确率的函数
def calculate_accuracy(outputs, targets):
    """
    计算分类准确率
    
    参数:
        outputs: 模型输出，形状为(batch_size, num_classes)
        targets: 真实标签，形状为(batch_size)
        
    返回:
        accuracy: 浮点数，表示准确率
    """
    # 获取每个样本的预测类别（最大值的索引）
    _, predicted = torch.max(outputs, 1)
    # 计算正确预测的数量
    correct = (predicted == targets).sum().item()
    # 计算准确率
    accuracy = correct / targets.size(0)
    return accuracy

# 定义获取设备的函数
def get_device():
    """
    获取可用的设备（GPU或CPU）
    
    返回:
        device: torch.device对象，表示可用的设备
    """
    # 检查是否支持MPS（Apple Silicon GPU，如Mac Pro 14的M系列芯片）
    if torch.backends.mps.is_available():
        return torch.device('mps')
    # 检查是否支持CUDA（NVIDIA GPU）
    elif torch.cuda.is_available():
        return torch.device('cuda')
    # 否则使用CPU
    else:
        return torch.device('cpu')