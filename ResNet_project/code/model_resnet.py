# 导入PyTorch相关库
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入常用的函数

# 定义BasicBlock，用于构建ResNet-18和ResNet-34
class BasicBlock(nn.Module):
    # expansion表示残差块输出通道数与输入通道数的比例
    # BasicBlock的expansion为1，因为输出通道数与输入通道数相同
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化BasicBlock
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长，默认为1
            downsample: 下采样模块，用于调整输入维度与输出维度匹配
        """
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层: 3x3卷积，stride由参数指定，padding=1保持尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，inplace=True表示原地操作，节省内存
        
        # 第二个卷积层: 3x3卷积，stride=1，padding=1保持尺寸
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批归一化层
        
        self.downsample = downsample  # 下采样模块，用于捷径连接
        self.stride = stride  # 记录步长
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图
        
        返回:
            经过残差块处理后的特征图
        """
        # 保存输入，用于捷径连接
        identity = x
        
        # 主分支: 卷积 -> 批归一化 -> ReLU -> 卷积 -> 批归一化
        out = self.conv1(x)  # 第一个卷积层
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        
        out = self.conv2(out)  # 第二个卷积层
        out = self.bn2(out)  # 批归一化
        
        # 如果需要下采样，对输入进行下采样，使维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接: 主分支输出 + 捷径连接
        out += identity
        out = self.relu(out)  # 最后再经过一次ReLU激活
        
        return out

# 定义BottleneckBlock，用于构建ResNet-50、ResNet-101和ResNet-152
class BottleneckBlock(nn.Module):
    # expansion表示残差块输出通道数与输入通道数的比例
    # BottleneckBlock的expansion为4，因为输出通道数是输入通道数的4倍
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化BottleneckBlock
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 卷积步长，默认为1
            downsample: 下采样模块，用于调整输入维度与输出维度匹配
        """
        super(BottleneckBlock, self).__init__()
        
        # 第一个卷积层: 1x1卷积，用于降维
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化层
        
        # 第二个卷积层: 3x3卷积，stride由参数指定，padding=1保持尺寸
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)  # 批归一化层
        
        # 第三个卷积层: 1x1卷积，用于升维
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  # 批归一化层
        
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.downsample = downsample  # 下采样模块
        self.stride = stride  # 记录步长
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征图
        
        返回:
            经过残差块处理后的特征图
        """
        # 保存输入，用于捷径连接
        identity = x
        
        # 主分支: 1x1卷积(降维) -> 批归一化 -> ReLU -> 3x3卷积 -> 批归一化 -> ReLU -> 1x1卷积(升维) -> 批归一化
        out = self.conv1(x)  # 1x1卷积降维
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        
        out = self.conv2(out)  # 3x3卷积
        out = self.bn2(out)  # 批归一化
        out = self.relu(out)  # ReLU激活
        
        out = self.conv3(out)  # 1x1卷积升维
        out = self.bn3(out)  # 批归一化
        
        # 如果需要下采样，对输入进行下采样，使维度匹配
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 残差连接: 主分支输出 + 捷径连接
        out += identity
        out = self.relu(out)  # 最后再经过一次ReLU激活
        
        return out

# 定义ResNet主类
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        """
        初始化ResNet
        
        参数:
            block: 残差块类型(BasicBlock或BottleneckBlock)
            layers: 每个阶段的残差块数量
            num_classes: 分类类别数，默认为1000
            zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
        """
        super(ResNet, self).__init__()
        
        # 初始输入通道数
        self.in_channels = 64
        
        # 第一个卷积层: 7x7卷积，stride=2，padding=3，将输入尺寸从224x224降到112x112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)  # 批归一化层
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        
        # 最大池化层: 3x3池化，stride=2，padding=1，将尺寸从112x112降到56x56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建四个阶段的残差层
        # layer1: 输出通道数64，不改变尺寸(56x56)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # layer2: 输出通道数128，尺寸减半(28x28)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # layer3: 输出通道数256，尺寸减半(14x14)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # layer4: 输出通道数512，尺寸减半(7x7)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 自适应平均池化: 将任意尺寸的特征图池化为1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层: 将特征映射到类别数
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态分布初始化卷积层权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 将残差分支的最后一个BN层初始化为0，这有助于模型收敛
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckBlock):
                    nn.init.constant_(m.bn3.weight, 0)  # BottleneckBlock的最后一个BN层
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # BasicBlock的最后一个BN层
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建一个阶段的残差层
        
        参数:
            block: 残差块类型
            out_channels: 输出通道数
            blocks: 该阶段的残差块数量
            stride: 步长，默认为1
        
        返回:
            由多个残差块组成的Sequential模块
        """
        downsample = None
        
        # 如果步长不为1，或者输入通道数不等于输出通道数*expansion，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # 下采样模块: 1x1卷积 + 批归一化
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        # 添加第一个残差块，可能包含下采样
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        # 更新输入通道数，用于下一个残差块
        self.in_channels = out_channels * block.expansion
        
        # 添加剩余的残差块，步长为1，不需要下采样
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        # 返回由残差块组成的Sequential模块
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入图像，形状为(batch_size, 3, 224, 224)
        
        返回:
            分类结果，形状为(batch_size, num_classes)
        """
        # 输入 -> 卷积 -> 批归一化 -> ReLU -> 最大池化
        x = self.conv1(x)  # 7x7卷积
        x = self.bn1(x)  # 批归一化
        x = self.relu(x)  # ReLU激活
        x = self.maxpool(x)  # 最大池化
        
        # 四个阶段的残差层
        x = self.layer1(x)  # 第一阶段
        x = self.layer2(x)  # 第二阶段
        x = self.layer3(x)  # 第三阶段
        x = self.layer4(x)  # 第四阶段
        
        # 自适应平均池化 -> 展平 -> 全连接层
        x = self.avgpool(x)  # 自适应平均池化，输出1x1
        x = torch.flatten(x, 1)  # 展平为一维向量
        x = self.fc(x)  # 全连接层，输出分类结果
        
        return x

# 定义ResNet-18
def resnet18(num_classes=1000, zero_init_residual=False):
    """
    构建ResNet-18模型
    
    参数:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
    
    返回:
        ResNet-18模型
    """
    # 使用BasicBlock，每个阶段的残差块数量为[2, 2, 2, 2]
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, zero_init_residual=zero_init_residual)

# 定义ResNet-34
def resnet34(num_classes=1000, zero_init_residual=False):
    """
    构建ResNet-34模型
    
    参数:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
    
    返回:
        ResNet-34模型
    """
    # 使用BasicBlock，每个阶段的残差块数量为[3, 4, 6, 3]
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, zero_init_residual=zero_init_residual)

# 定义ResNet-50
def resnet50(num_classes=1000, zero_init_residual=False):
    """
    构建ResNet-50模型
    
    参数:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
    
    返回:
        ResNet-50模型
    """
    # 使用BottleneckBlock，每个阶段的残差块数量为[3, 4, 6, 3]
    return ResNet(BottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, zero_init_residual=zero_init_residual)

# 定义ResNet-101
def resnet101(num_classes=1000, zero_init_residual=False):
    """
    构建ResNet-101模型
    
    参数:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
    
    返回:
        ResNet-101模型
    """
    # 使用BottleneckBlock，每个阶段的残差块数量为[3, 4, 23, 3]
    return ResNet(BottleneckBlock, [3, 4, 23, 3], num_classes=num_classes, zero_init_residual=zero_init_residual)

# 定义ResNet-152
def resnet152(num_classes=1000, zero_init_residual=False):
    """
    构建ResNet-152模型
    
    参数:
        num_classes: 分类类别数
        zero_init_residual: 是否将残差分支的最后一个BN层初始化为0
    
    返回:
        ResNet-152模型
    """
    # 使用BottleneckBlock，每个阶段的残差块数量为[3, 8, 36, 3]
    return ResNet(BottleneckBlock, [3, 8, 36, 3], num_classes=num_classes, zero_init_residual=zero_init_residual)