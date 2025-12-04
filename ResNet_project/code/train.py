# 导入必要的库
import torch  # PyTorch核心库
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
import time  # 用于计时
import os  # 文件路径操作
# 从dataset.py导入数据集相关函数
from dataset import get_datasets, get_data_loaders
# 从utils.py导入工具函数
from utils import (
    setup_logging,      # 设置日志
    get_model,          # 获取模型
    get_optimizer,      # 获取优化器
    get_scheduler,      # 获取学习率调度器
    save_model,         # 保存模型
    calculate_accuracy, # 计算准确率
    get_device          # 获取设备
)
# 从config.py导入配置
from config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, EXPERIMENTS, OUTPUT_CONFIG

# 定义训练一个epoch的函数
def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """
    训练一个epoch
    
    参数:
        model: 模型对象
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备（GPU或CPU）
        logger: 日志记录器
        
    返回:
        train_loss: 训练损失
        train_acc: 训练准确率
    """
    # 设置模型为训练模式
    model.train()
    # 初始化训练损失和正确预测数量
    train_loss = 0.0
    train_correct = 0
    total = 0
    
    # 遍历训练数据加载器
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # 将数据移动到指定设备
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播：计算模型输出
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新模型参数
        
        # 统计损失和准确率
        train_loss += loss.item()  # 累加损失
        _, predicted = outputs.max(1)  # 获取预测类别
        total += targets.size(0)  # 累加总样本数
        train_correct += predicted.eq(targets).sum().item()  # 累加正确预测数
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {100.*train_correct/total:.2f}%')
    
    # 计算平均损失和准确率
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / total
    
    return train_loss, train_acc

# 定义验证一个epoch的函数
def validate_epoch(model, val_loader, criterion, device):
    """
    验证一个epoch
    
    参数:
        model: 模型对象
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备（GPU或CPU）
        
    返回:
        val_loss: 验证损失
        val_acc: 验证准确率
    """
    # 设置模型为评估模式
    model.eval()
    # 初始化验证损失和正确预测数量
    val_loss = 0.0
    val_correct = 0
    total = 0
    
    # 关闭梯度计算，节省内存和计算资源
    with torch.no_grad():
        # 遍历验证数据加载器
        for inputs, targets in val_loader:
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播：计算模型输出
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 统计损失和准确率
            val_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测类别
            total += targets.size(0)  # 累加总样本数
            val_correct += predicted.eq(targets).sum().item()  # 累加正确预测数
    
    # 计算平均损失和准确率
    val_loss /= len(val_loader)
    val_acc = 100. * val_correct / total
    
    return val_loss, val_acc

# 定义训练模型的函数
def train_model(config, experiment_name):
    """
    训练模型
    
    参数:
        config: 实验配置字典
        experiment_name: 实验名称
        
    返回:
        best_acc: 最佳验证准确率
    """
    # 设置日志记录
    logger = setup_logging(OUTPUT_CONFIG['log_dir'], experiment_name)
    logger.info(f'Starting experiment: {experiment_name}')
    logger.info(f'Config: {config}')
    
    # 获取可用设备
    device = get_device()
    logger.info(f'Using device: {device}')
    
    # 加载数据集
    logger.info('Loading datasets...')
    train_dataset, test_dataset = get_datasets(DATA_CONFIG['data_dir'], augment=config['augment'])
    train_loader, test_loader = get_data_loaders(
        train_dataset, 
        test_dataset, 
        batch_size=config['batch_size'], 
        num_workers=DATA_CONFIG['num_workers']
    )
    
    # 获取类别数量和样本数量
    num_classes = len(train_dataset.classes)
    logger.info(f'Number of classes: {num_classes}')
    logger.info(f'Training samples: {len(train_dataset)}')
    logger.info(f'Testing samples: {len(test_dataset)}')
    
    # 创建模型
    logger.info(f'Creating model: {config["model_name"]}')
    model = get_model(
        config['model_name'], 
        num_classes, 
        pretrained=config['pretrained'],
        zero_init_residual=MODEL_CONFIG['zero_init_residual']
    )
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 设置损失函数：交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    # 设置优化器
    optimizer = get_optimizer(
        model, 
        config['optimizer'], 
        config['learning_rate'], 
        TRAIN_CONFIG['weight_decay']
    )
    
    # 设置学习率调度器
    scheduler = get_scheduler(
        optimizer, 
        TRAIN_CONFIG['scheduler'], 
        step_size=TRAIN_CONFIG['step_size'], 
        gamma=TRAIN_CONFIG['gamma']
    )
    
    # 训练循环初始化
    best_acc = 0.0  # 最佳验证准确率
    start_time = time.time()  # 开始时间
    
    # 遍历每个epoch
    for epoch in range(config['epochs']):
        logger.info(f'\nEpoch {epoch+1}/{config["epochs"]}')
        logger.info('-' * 50)
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        
        # 验证一个epoch
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
        
        # 更新学习率
        if TRAIN_CONFIG['scheduler'] == 'plateau':
            scheduler.step(val_acc)  # 基于验证准确率调整学习率
        else:
            scheduler.step()  # 按计划调整学习率
        
        # 记录结果
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, OUTPUT_CONFIG['model_dir'], experiment_name, epoch+1)
            logger.info(f'Best model saved with accuracy: {best_acc:.2f}%')
        
        # 定期保存模型
        if (epoch + 1) % OUTPUT_CONFIG['save_freq'] == 0:
            save_model(model, OUTPUT_CONFIG['model_dir'], experiment_name, epoch+1)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    logger.info(f'\nTraining completed!')
    logger.info(f'Total training time: {total_time:.2f} seconds')
    logger.info(f'Best test accuracy: {best_acc:.2f}%')
    
    # 返回最佳准确率、训练轮数和时间
    return best_acc, config['epochs'], total_time

# 主函数
def main():
    """
    主函数，运行所有实验
    """
    # 初始化结果字典
    results = {}
    
    # 遍历所有实验配置
    for experiment_name, config in EXPERIMENTS.items():
        # 设置日志记录器
        logger = setup_logging(OUTPUT_CONFIG['log_dir'], f'main')
        logger.info(f'\n' + '='*60)
        logger.info(f'Running experiment: {experiment_name}')
        logger.info(f'='*60)
        
        # 训练模型
        best_acc, epochs, total_time = train_model(config, experiment_name)
        # 保存实验结果
        results[experiment_name] = {"accuracy": best_acc, "epochs": epochs, "time": total_time}
    
    # 打印所有实验结果
    logger.info(f'\n' + '='*60)
    logger.info(f'All experiments completed!')
    logger.info(f'='*60)
    logger.info(f'Experiment Results:')
    for experiment_name, result in results.items():
        logger.info(f'{experiment_name}: {result["accuracy"]:.2f}% - epochs: {result["epochs"]} - time: {result["time"]:.2f} seconds')
    
    # 保存结果到文件
    results_file = os.path.join(OUTPUT_CONFIG['log_dir'], 'results.txt')
    with open(results_file, 'w') as f:
        for experiment_name, result in results.items():
            f.write(f'{experiment_name}: {result["accuracy"]:.2f}% - epochs: {result["epochs"]} - time: {result["time"]:.2f} seconds\n')
    
    logger.info(f'Results saved to: {results_file}')

# 如果直接运行该脚本，执行主函数
if __name__ == '__main__':
    main()