# 分布式PlaneRCNN系统设计方案

## 1. 系统架构

### 1.1 核心架构

```
┌─────────────────┐          ┌─────────────────┐
│   树莓派小车     │◄────────►│      电脑       │
│  (边缘设备)     │ 网络传输  │ (计算中心)     │
└─────────────────┘          └─────────────────┘
        │                          │
        ▼                          ▼
┌─────────────────┐          ┌─────────────────┐
│  摄像头采集图像  │          │ PlaneRCNN处理  │
└─────────────────┘          └─────────────────┘
        │                          │
        │                          ▼
        │                  ┌─────────────────┐
        │                  │ 平面检测结果    │
        │                  └─────────────────┘
        │                          │
        │                          ▼
        └─────────────────►│ 投影控制指令    │
                            └─────────────────┘
```

### 1.2 功能划分

#### 树莓派端

* **图像采集**：使用摄像头实时采集图像

* **图像传输**：将图像通过网络传输到电脑

* **指令接收**：接收电脑发送的控制指令

* **电机控制**：根据指令控制小车运动

* **投影控制**：根据指令控制投影机

#### 电脑端

* **图像接收**：接收树莓派传输的图像

* **PlaneRCNN处理**：运行简化版或轻量级PlaneRCNN模型

* **结果分析**：分析检测到的平面，确定合适的投影位置

* **指令发送**：向树莓派发送控制指令

* **可视化显示**：实时显示检测结果

## 2. 项目文件结构

```
SphereMate/
├── plane_rcnn/                # PlaneRCNN平面检测模块
│   ├── model/                # 模型定义
│   │   ├── resnet.py          # ResNet主干网络
│   │   ├── fpn.py             # FPN特征金字塔
│   │   └── plane_rcnn.py      # PlaneRCNN主架构
│   ├── inference/             # 推理模块
│   │   └── detector.py        # 平面检测器
│   ├── projection/            # 投影模块
│   │   ├── projector.py       # 投影系统接口
│   │   ├── calibration.py     # 相机-投影机标定
│   │   └── visualizer.py      # 平面可视化
│   └── utils/                 # 工具类
│       └── camera.py          # 摄像头接口
├── raspberry_pi/              # 树莓派控制模块
│   ├── camera/               # 摄像头相关
│   │   └── camera_stream.py   # 图像采集与传输
│   ├── motor/                 # 电机控制
│   │   ├── motor_control.py   # 电机驱动
│   │   ├── pid_controller.py  # PID控制器
│   │   └── servo_control.py   # 舵机控制
│   ├── communication/         # 通信模块
│   │   ├── client.py          # 网络客户端
│   │   └── command_handler.py # 指令处理
│   └── main.py                # 树莓派主程序
├── common/                    # 公共模块
│   ├── config.py              # 配置文件
│   ├── utils.py               # 通用工具
│   └── protocols.py           # 通信协议
├── scripts/                   # 脚本文件
│   ├── remote_plane_rcnn.py   # 远程PlaneRCNN处理
│   └── server.py              # 服务器主程序
└── main.py                    # 主应用入口
```

## 3. 树莓派控制代码实现

### 3.1 电机控制模块

```python
# raspberry_pi/motor/motor_control.py
import RPi.GPIO as GPIO
import time

class MotorController:
    """电机控制器"""
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        
        # 电机引脚配置
        self.left_motor_pins = {
            'enable': self.config.get('left_motor_enable', 12),
            'in1': self.config.get('left_motor_in1', 11),
            'in2': self.config.get('left_motor_in2', 13)
        }
        
        self.right_motor_pins = {
            'enable': self.config.get('right_motor_enable', 16),
            'in1': self.config.get('right_motor_in1', 15),
            'in2': self.config.get('right_motor_in2', 18)
        }
        
        # 初始化GPIO
        GPIO.setmode(GPIO.BOARD)
        
        # 设置电机引脚为输出
        for pin in self.left_motor_pins.values():
            GPIO.setup(pin, GPIO.OUT)
        
        for pin in self.right_motor_pins.values():
            GPIO.setup(pin, GPIO.OUT)
        
        # 创建PWM对象
        self.left_pwm = GPIO.PWM(self.left_motor_pins['enable'], 1000)
        self.right_pwm = GPIO.PWM(self.right_motor_pins['enable'], 1000)
        
        # 启动PWM
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
        self.current_speed = 0
    
    def set_motor_direction(self, motor, direction):
        """设置电机方向"""
        if motor == 'left':
            pins = self.left_motor_pins
        else:
            pins = self.right_motor_pins
        
        if direction == 'forward':
            GPIO.output(pins['in1'], GPIO.HIGH)
            GPIO.output(pins['in2'], GPIO.LOW)
        elif direction == 'backward':
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.HIGH)
        else:
            GPIO.output(pins['in1'], GPIO.LOW)
            GPIO.output(pins['in2'], GPIO.LOW)
    
    def set_speed(self, speed):
        """设置电机速度"""
        # 限制速度范围
        speed = max(-100, min(100, speed))
        self.current_speed = speed
        
        if speed > 0:
            # 前进
            self.set_motor_direction('left', 'forward')
            self.set_motor_direction('right', 'forward')
            self.left_pwm.ChangeDutyCycle(speed)
            self.right_pwm.ChangeDutyCycle(speed)
        elif speed < 0:
            # 后退
            self.set_motor_direction('left', 'backward')
            self.set_motor_direction('right', 'backward')
            self.left_pwm.ChangeDutyCycle(abs(speed))
            self.right_pwm.ChangeDutyCycle(abs(speed))
        else:
            # 停止
            self.left_pwm.ChangeDutyCycle(0)
            self.right_pwm.ChangeDutyCycle(0)
    
    def turn(self, angle):
        """转向控制"""
        # 简单的转向实现，根据角度调整左右电机速度
        if angle > 0:
            # 右转
            self.set_motor_direction('left', 'forward')
            self.set_motor_direction('right', 'backward')
        elif angle < 0:
            # 左转
            self.set_motor_direction('left', 'backward')
            self.set_motor_direction('right', 'forward')
        else:
            # 直行
            self.set_motor_direction('left', 'forward')
            self.set_motor_direction('right', 'forward')
        
        # 根据角度调整速度
        turn_speed = abs(angle) * 0.8
        self.left_pwm.ChangeDutyCycle(turn_speed)
        self.right_pwm.ChangeDutyCycle(turn_speed)
    
    def stop(self):
        """停止电机"""
        self.set_speed(0)
    
    def cleanup(self):
        """清理资源"""
        self.stop()
        self.left_pwm.stop()
        self.right_pwm.stop()
        GPIO.cleanup()
```

### 3.2 PID控制器

```python
# raspberry_pi/motor/pid_controller.py
class PIDController:
    """PID控制器"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.0):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        
        self.last_error = 0.0
        self.integral = 0.0
        self.dt = 0.01  # 默认时间间隔
    
    def update(self, setpoint, process_value):
        """更新PID控制器
        
        Args:
            setpoint: 目标值
            process_value: 当前值
            
        Returns:
            float: 控制输出
        """
        # 计算误差
        error = setpoint - process_value
        
        # 计算积分项
        self.integral += error * self.dt
        
        # 计算微分项
        derivative = (error - self.last_error) / self.dt
        
        # 计算PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # 更新最后误差
        self.last_error = error
        
        return output
    
    def reset(self):
        """重置PID控制器"""
        self.last_error = 0.0
        self.integral = 0.0
    
    def set_gains(self, kp, ki, kd):
        """设置PID增益"""
        self.kp = kp
        self.ki = ki
        self.kd = kd
```

### 3.3 图像采集与传输

```python
# raspberry_pi/camera/camera_stream.py
import cv2
import socket
import numpy as np
import threading

class CameraStream:
    """摄像头图像采集与传输"""
    def __init__(self, camera_id=0, resolution=(640, 480), fps=30):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 通信相关
        self.client_socket = None
        self.is_streaming = False
        self.stream_thread = None
        
        # 图像质量设置
        self.jpeg_quality = 80
    
    def connect(self, server_ip, server_port):
        """连接到服务器"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((server_ip, server_port))
            print(f"已连接到服务器 {server_ip}:{server_port}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def start_streaming(self):
        """开始图像传输"""
        if self.is_streaming:
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        print("图像传输已启动")
    
    def _stream_loop(self):
        """图像传输循环"""
        while self.is_streaming and self.client_socket:
            try:
                # 读取图像
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # 编码图像
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                if not result:
                    continue
                
                # 将图像转换为字节流
                data = encimg.tobytes()
                
                # 发送数据长度
                data_len = len(data).to_bytes(4, byteorder='big')
                self.client_socket.sendall(data_len)
                
                # 发送图像数据
                self.client_socket.sendall(data)
                
            except Exception as e:
                print(f"传输错误: {e}")
                self.is_streaming = False
                break
    
    def stop_streaming(self):
        """停止图像传输"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        print("图像传输已停止")
    
    def get_frame(self):
        """获取单帧图像"""
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        """释放资源"""
        self.stop_streaming()
        if self.client_socket:
            self.client_socket.close()
        if self.cap:
            self.cap.release()
        print("摄像头资源已释放")
```

### 3.4 网络客户端

```python
# raspberry_pi/communication/client.py
import socket
import json
import threading

class NetworkClient:
    """网络客户端"""
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.is_connected = False
        self.receive_thread = None
        self.command_callback = None
    
    def connect(self):
        """连接到服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))
            self.is_connected = True
            print(f"已连接到服务器 {self.server_ip}:{self.server_port}")
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False
    
    def _receive_loop(self):
        """接收消息循环"""
        while self.is_connected and self.socket:
            try:
                # 接收数据
                data = self.socket.recv(1024)
                if not data:
                    break
                
                # 解析JSON数据
                message = json.loads(data.decode())
                
                # 处理消息
                if self.command_callback:
                    self.command_callback(message)
                
            except Exception as e:
                print(f"接收错误: {e}")
                break
        
        self.is_connected = False
    
    def send_message(self, message):
        """发送消息"""
        if not self.is_connected or not self.socket:
            return False
        
        try:
            # 发送JSON数据
            data = json.dumps(message).encode()
            self.socket.sendall(data)
            return True
        except Exception as e:
            print(f"发送错误: {e}")
            return False
    
    def set_command_callback(self, callback):
        """设置指令回调函数"""
        self.command_callback = callback
    
    def disconnect(self):
        """断开连接"""
        self.is_connected = False
        if self.socket:
            self.socket.close()
        print("已断开与服务器的连接")
```

### 3.5 树莓派主程序

```python
# raspberry_pi/main.py
import sys
import time
from raspberry_pi.camera.camera_stream import CameraStream
from raspberry_pi.motor.motor_control import MotorController
from raspberry_pi.motor.pid_controller import PIDController
from raspberry_pi.communication.client import NetworkClient

class SphereMateRaspberryPi:
    """SphereMate树莓派主程序"""
    def __init__(self, config):
        self.config = config
        
        # 初始化模块
        self.camera = CameraStream(
            camera_id=config['camera_id'],
            resolution=config['resolution'],
            fps=config['fps']
        )
        
        self.motor_controller = MotorController(config.get('motor', {}))
        self.pid_controller = PIDController(**config.get('pid', {}))
        
        self.network_client = NetworkClient(
            server_ip=config['server_ip'],
            server_port=config['server_port']
        )
        
        # 注册指令回调
        self.network_client.set_command_callback(self.handle_command)
    
    def handle_command(self, command):
        """处理来自服务器的指令"""
        print(f"收到指令: {command}")
        
        command_type = command.get('type')
        if command_type == 'move':
            params = command.get('params', {})
            speed = params.get('speed', 0)
            angle = params.get('angle', 0)
            self.motor_controller.set_speed(speed)
            if angle != 0:
                self.motor_controller.turn(angle)
        
        elif command_type == 'stop':
            self.motor_controller.stop()
        
        elif command_type == 'project':
            # 投影控制逻辑
            pass
        
        elif command_type == 'calibrate':
            # 标定逻辑
            pass
    
    def start(self):
        """启动系统"""
        print("SphereMate树莓派系统启动中...")
        
        # 连接到服务器
        if not self.network_client.connect():
            print("无法连接到服务器，系统启动失败")
            return False
        
        # 开始图像传输
        self.camera.start_streaming()
        
        print("SphereMate树莓派系统已启动")
        return True
    
    def run(self):
        """主运行循环"""
        if not self.start():
            return
        
        try:
            while True:
                # 主循环，处理系统状态
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n系统正在关闭...")
        finally:
            self.stop()
    
    def stop(self):
        """停止系统"""
        # 停止图像传输
        self.camera.stop_streaming()
        
        # 停止电机
        self.motor_controller.stop()
        
        # 断开网络连接
        self.network_client.disconnect()
        
        # 释放资源
        self.camera.release()
        self.motor_controller.cleanup()
        
        print("SphereMate树莓派系统已关闭")

if __name__ == "__main__":
    # 配置文件
    config = {
        'camera_id': 0,
        'resolution': (640, 480),
        'fps': 30,
        'server_ip': '192.168.1.100',
        'server_port': 5000,
        'motor': {
            'left_motor_enable': 12,
            'left_motor_in1': 11,
            'left_motor_in2': 13,
            'right_motor_enable': 16,
            'right_motor_in1': 15,
            'right_motor_in2': 18
        },
        'pid': {
            'kp': 1.0,
            'ki': 0.0,
            'kd': 0.1
        }
    }
    
    # 创建并运行系统
    sphere_mate = SphereMateRaspberryPi(config)
    sphere_mate.run()
```

## 4. 服务器端代码

```python
# scripts/remote_plane_rcnn.py
import socket
import threading
import cv2
import numpy as np
from plane_rcnn.inference.detector import PlaneDetector

class RemotePlaneRCNNServer:
    """远程PlaneRCNN服务器"""
    def __init__(self, port=5000):
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('0.0.0.0', port))
        self.server_socket.listen(5)
        
        self.detector = PlaneDetector(use_lightweight=False)
        self.clients = []
        
        print(f"远程PlaneRCNN服务器已启动，监听端口 {port}")
    
    def handle_client(self, client_socket, client_addr):
        """处理客户端连接"""
        print(f"客户端已连接: {client_addr}")
        self.clients.append(client_socket)
        
        try:
            while True:
                # 接收图像数据大小
                data_len_bytes = client_socket.recv(4)
                if not data_len_bytes:
                    break
                
                data_len = int.from_bytes(data_len_bytes, byteorder='big')
                
                # 接收图像数据
                image_data = b''
                while len(image_data) < data_len:
                    packet = client_socket.recv(data_len - len(image_data))
                    if not packet:
                        break
                    image_data += packet
                
                if len(image_data) != data_len:
                    break
                
                # 解码图像
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # 进行平面检测
                results = self.detector.detect(frame)
                
                # 可视化结果
                vis_frame = self.detector.visualize_detections(frame, results)
                
                # 显示结果
                cv2.imshow('Remote Plane Detection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # 发送检测结果（简化版）
                result_msg = {
                    'type': 'detection_result',
                    'params': {
                        'num_planes': len(results['planes']),
                        'planes': [
                            {
                                'confidence': plane['confidence'],
                                'bbox': plane['bbox'].tolist()
                            } for plane in results['planes']
                        ]
                    }
                }
                
        except Exception as e:
            print(f"处理客户端 {client_addr} 时出错: {e}")
        finally:
            client_socket.close()
            self.clients.remove(client_socket)
            print(f"客户端 {client_addr} 已断开连接")
    
    def run(self):
        """运行服务器"""
        try:
            while True:
                client_socket, client_addr = self.server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, client_addr)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\n服务器正在关闭...")
        finally:
            self.stop()
    
    def stop(self):
        """停止服务器"""
        for client in self.clients:
            client.close()
        self.server_socket.close()
        cv2.destroyAllWindows()
        print("服务器已关闭")

if __name__ == "__main__":
    server = RemotePlaneRCNNServer()
    server.run()
```

## 5. 配置文件

```python
# common/config.py
# 系统配置文件

# PlaneRCNN配置
PLANE_RCNN_CONFIG = {
    'device': 'cpu',
    'backbone_type': 'resnet18',
    'fpn_type': 'lightweight',
    'input_size': (160, 160),
    'use_quantization': False
}

# 网络配置
NETWORK_CONFIG = {
    'server_ip': '192.168.1.100',
    'server_port': 5000,
    'buffer_size': 1024,
    'timeout': 5
}

# 摄像头配置
CAMERA_CONFIG = {
    'camera_id': 0,
    'resolution': (640, 480),
    'fps': 30,
    'jpeg_quality': 80
}

# 电机配置
MOTOR_CONFIG = {
    'left_motor': {
        'enable_pin': 12,
        'in1_pin': 11,
        'in2_pin': 13
    },
    'right_motor': {
        'enable_pin': 16,
        'in1_pin': 15,
        'in2_pin': 18
    },
    'max_speed': 100
}

# PID配置
PID_CONFIG = {
    'kp': 1.0,
    'ki': 0.0,
    'kd': 0.1
}
```

## 6. 使用方法

### 6.1 电脑端运行

```bash
# 启动远程PlaneRCNN服务器
python scripts/remote_plane_rcnn.py
```

### 6.2 树莓派端运行

```bash
# 运行树莓派主程序
python raspberry_pi/main.py
```

### 6.3 配置修改

所有配置都集中在 `common/config.py` 文件中，可以根据实际情况修改。

## 7. 系统工作流程

1. **启动服务器**：在电脑上运行 `remote_plane_rcnn.py`，启动PlaneRCNN服务器
2. **启动树莓派**：在树莓派上运行 `main.py`，连接到服务器并开始图像传输
3. **图像处理**：服务器接收树莓派传输的图像，运行PlaneRCNN进行平面检测
4. **结果显示**：服务器显示检测结果，同时将结果发送回树莓派
5. **控制指令**：服务器可以发送控制指令给树莓派，控制小车运动和投影

## 8. 扩展功能

* **自动导航**：基于检测到的平面，实现小车的自动导航
* **多平面投影**：在多个平面上同时投影内容
* **增强现实**：在检测到的平面上叠加虚拟内容
* **远程控制界面**：创建Web界面，实现远程监控和控制

## 9. 技术栈

* **电脑端**：Python, PyTorch, OpenCV, Socket
* **树莓派**：Python, OpenCV, RPi.GPIO, Socket
* **通信协议**：TCP/IP, JSON
* **操作系统**：Windows/macOS (电脑), Raspberry Pi OS (树莓派)

这个设计方案整合了PlaneRCNN平面检测和树莓派控制代码，实现了一个完整的分布式系统，能够满足用户的需求：在电脑上运行PlaneRCNN，树莓派负责图像采集和运动控制。