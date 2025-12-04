class MotorController:
    """电机控制器
    
    用于控制电机的预留接口
    """
    def __init__(self, config=None):
        """初始化电机控制器
        
        Args:
            config: 电机配置
        """
        self.config = config if config is not None else {}
        self.is_connected = False
        
        # 电机参数
        self.motor_ids = self.config.get('motor_ids', [1, 2])
        self.max_speed = self.config.get('max_speed', 100)
    
    def connect(self):
        """连接到电机控制器
        
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        try:
            print("正在连接到电机控制器...")
            # 实际应用中，这里会包含与电机控制器的通信代码
            self.is_connected = True
            print("电机控制器连接成功")
            return True
        except Exception as e:
            print(f"连接电机控制器失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开与电机控制器的连接
        
        Returns:
            bool: 断开成功返回True，否则返回False
        """
        try:
            print("正在断开电机控制器连接...")
            # 实际应用中，这里会包含与电机控制器的通信代码
            self.is_connected = False
            print("电机控制器断开成功")
            return True
        except Exception as e:
            print(f"断开电机控制器连接失败: {e}")
            return False
    
    def set_motor_speed(self, motor_id, speed):
        """设置电机速度
        
        Args:
            motor_id: 电机ID
            speed: 速度值(-max_speed到max_speed)
            
        Returns:
            bool: 设置成功返回True，否则返回False
        """
        if not self.is_connected:
            print("错误：电机控制器未连接")
            return False
        
        try:
            # 限制速度范围
            speed = max(-self.max_speed, min(self.max_speed, speed))
            print(f"设置电机{motor_id}速度为: {speed}")
            # 实际应用中，这里会包含与电机控制器的通信代码
            return True
        except Exception as e:
            print(f"设置电机速度失败: {e}")
            return False
    
    def stop_motor(self, motor_id):
        """停止电机
        
        Args:
            motor_id: 电机ID
            
        Returns:
            bool: 停止成功返回True，否则返回False
        """
        return self.set_motor_speed(motor_id, 0)
    
    def stop_all_motors(self):
        """停止所有电机
        
        Returns:
            bool: 停止成功返回True，否则返回False
        """
        success = True
        for motor_id in self.motor_ids:
            if not self.stop_motor(motor_id):
                success = False
        return success
    
    def get_motor_status(self, motor_id):
        """获取电机状态
        
        Args:
            motor_id: 电机ID
            
        Returns:
            dict: 电机状态字典
        """
        return {
            'motor_id': motor_id,
            'speed': 0,  # 模拟速度值
            'temperature': 30,  # 模拟温度值
            'status': 'normal'
        }
    
    def get_all_motors_status(self):
        """获取所有电机状态
        
        Returns:
            dict: 所有电机状态字典
        """
        status = {}
        for motor_id in self.motor_ids:
            status[motor_id] = self.get_motor_status(motor_id)
        return status