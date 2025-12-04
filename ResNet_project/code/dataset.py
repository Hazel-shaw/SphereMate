# 导入必要的库
import os  # 用于文件路径操作
import torch  # PyTorch核心库
from torchvision import datasets, transforms  # 用于数据集加载和数据转换
from torch.utils.data import DataLoader  # 用于创建数据加载器

# 定义获取数据转换的函数
def get_data_transforms(augment=False):
    """
    获取训练集和测试集的数据转换
    
    参数:
        augment: 布尔值，表示是否对训练集使用数据增强
        
    返回:
        train_transform: 训练集的数据转换
        test_transform: 测试集的数据转换
    """
    # 基础转换列表，适用于所有数据集
    base_transform = [
        # 将图像调整为224x224大小，这是ResNet模型的输入要求
        transforms.Resize((224, 224)),
        # 将PIL图像转换为PyTorch张量，范围从[0, 255]转换为[0, 1]
        transforms.ToTensor(),
        # 对图像进行归一化，使用ImageNet数据集的均值和标准差
        # 这有助于模型更快收敛
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    # 如果需要数据增强
    if augment:
        # 训练集转换：包含多种数据增强技术
        train_transform = transforms.Compose([
            # 随机裁剪并调整大小为224x224
            transforms.RandomResizedCrop(224),
            # 随机水平翻转，概率为0.5
            transforms.RandomHorizontalFlip(),
            # 随机垂直翻转，概率为0.5
            transforms.RandomVerticalFlip(),
            # 随机调整亮度、对比度、饱和度和色调
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # 随机旋转图像，角度范围为[-15, 15]度
            transforms.RandomRotation(15),
            # 转换为张量
            transforms.ToTensor(),
            # 归一化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # 不使用数据增强时，训练集只使用基础转换
        train_transform = transforms.Compose(base_transform)
    
    # 测试集始终只使用基础转换，不进行数据增强
    test_transform = transforms.Compose(base_transform)
    
    return train_transform, test_transform

# 定义获取数据集的函数
def get_datasets(data_dir, augment=False):
    """
    获取训练集和测试集
    
    参数:
        data_dir: 字符串，表示数据集的根目录
        augment: 布尔值，表示是否对训练集使用数据增强
        
    返回:
        train_dataset: 训练集对象
        test_dataset: 测试集对象
    """
    # 获取数据转换
    train_transform, test_transform = get_data_transforms(augment)
    
    # 构建训练集和测试集的目录路径
    # 假设数据集目录结构为：data_dir/Training/类别1, data_dir/Training/类别2, ...
    # 以及 data_dir/Test/类别1, data_dir/Test/类别2, ...
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Test')
    
    # 使用ImageFolder加载数据集
    # ImageFolder会自动根据文件夹名称创建类别标签
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    return train_dataset, test_dataset

# 定义获取数据加载器的函数
def get_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    获取训练集和测试集的数据加载器
    
    参数:
        train_dataset: 训练集对象
        test_dataset: 测试集对象
        batch_size: 整数，表示每个批次的样本数量
        num_workers: 整数，表示用于加载数据的进程数量
        
    返回:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
    """
    # 创建训练集数据加载器
    train_loader = DataLoader(
        train_dataset,  # 训练集对象
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 训练集需要打乱顺序
        num_workers=num_workers,  # 工作进程数
        pin_memory=True  # 启用内存固定，加速数据传输到GPU
    )
    
    # 创建测试集数据加载器
    test_loader = DataLoader(
        test_dataset,  # 测试集对象
        batch_size=batch_size,  # 批次大小
        shuffle=False,  # 测试集不需要打乱顺序
        num_workers=num_workers,  # 工作进程数
        pin_memory=True  # 启用内存固定
    )
    
    return train_loader, test_loader

# 定义获取类别名称的函数
def get_class_names(dataset):
    """
    获取数据集中的类别名称列表
    
    参数:
        dataset: 数据集对象
        
    返回:
        类别名称列表
    """
    # ImageFolder数据集对象的classes属性包含了所有类别的名称
    return dataset.classes

# 定义获取类别数量的函数
def get_num_classes(dataset):
    """
    获取数据集中的类别数量
    
    参数:
        dataset: 数据集对象
        
    返回:
        类别数量
    """
    # 类别数量等于类别名称列表的长度
    return len(dataset.classes)