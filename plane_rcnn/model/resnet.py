import sys
import os
import torch
import torch.nn as nn

# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加现有ResNet项目路径
resnet_code_path = os.path.abspath(os.path.join(current_dir, '../../ResNet_project/code'))
sys.path.append(resnet_code_path)

# 导入resnet模型
from model_resnet import resnet18, resnet50

def get_resnet_backbone(pretrained=True, model_type='resnet18'):
    """获取轻量级ResNet主干网络，去除分类层
    
    Args:
        pretrained: 是否加载预训练权重
        model_type: 模型类型，可选'resnet18'或'resnet50'
        
    Returns:
        nn.Sequential: ResNet主干网络，用于特征提取
    """
    # 创建ResNet模型
    if model_type == 'resnet18':
        model = resnet18()
        model_name = 'resnet18'
    else:
        model = resnet50()
        model_name = 'resnet50'
    
    if pretrained:
        try:
            # 尝试加载现有模型权重
            model_path = os.path.abspath(os.path.join(current_dir, f'../../ResNet_project/models/config_a_epoch_1.pth'))
            model.load_state_dict(torch.load(model_path))
            print(f"成功加载{model_name}预训练权重")
        except Exception as e:
            print(f"加载{model_name}预训练权重失败: {e}")
            print("将使用随机初始化权重")
    
    # 去除分类层，保留特征提取部分
    # 输出特征图尺寸：输入224x224 -> 输出7x7
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

def get_resnet_features(model_type='resnet18'):
    """获取ResNet各层的输出特征
    
    Args:
        model_type: 模型类型，可选'resnet18'或'resnet50'
        
    Returns:
        dict: 包含各层输出特征的字典
    """
    # 创建ResNet模型
    if model_type == 'resnet18':
        model = resnet18()
    else:
        model = resnet50()
    
    # 定义各层特征提取
    features = {
        'conv1': nn.Sequential(model.conv1, model.bn1, model.relu),
        'maxpool': model.maxpool,
        'layer1': model.layer1,
        'layer2': model.layer2,
        'layer3': model.layer3,
        'layer4': model.layer4
    }
    
    return features

if __name__ == "__main__":
    # 测试ResNet-18主干网络加载
    backbone = get_resnet_backbone(pretrained=True, model_type='resnet18')
    print("ResNet-18主干网络结构:")
    print(backbone)
    
    # 测试特征提取
    input_tensor = torch.randn(1, 3, 224, 224)
    output = backbone(input_tensor)
    print(f"\n输入尺寸: {input_tensor.shape}")
    print(f"输出尺寸: {output.shape}")