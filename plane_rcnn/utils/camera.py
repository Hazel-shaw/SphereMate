import cv2
import numpy as np

class Camera:
    """摄像头接口
    
    用于图像捕获和摄像头控制
    """
    def __init__(self, config=None):
        """初始化摄像头
        
        Args:
            config: 摄像头配置
        """
        self.config = config if config is not None else {}
        self.cap = None
        self.is_opened = False
        
        # 摄像头参数
        self.camera_id = self.config.get('camera_id', 0)
        self.resolution = self.config.get('resolution', (640, 480))
        self.fps = self.config.get('fps', 30)
    
    def open(self):
        """打开摄像头
        
        Returns:
            bool: 打开成功返回True，否则返回False
        """
        try:
            # 打开摄像头
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 检查摄像头是否成功打开
            if self.cap.isOpened():
                self.is_opened = True
                print(f"摄像头{self.camera_id}已成功打开，分辨率: {self.resolution}")
                return True
            else:
                print(f"无法打开摄像头{self.camera_id}")
                self.is_opened = False
                return False
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            self.is_opened = False
            return False
    
    def close(self):
        """关闭摄像头
        
        Returns:
            bool: 关闭成功返回True，否则返回False
        """
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
                self.is_opened = False
                print(f"摄像头{self.camera_id}已关闭")
                return True
            else:
                print(f"摄像头{self.camera_id}未打开")
                return False
        except Exception as e:
            print(f"关闭摄像头失败: {e}")
            return False
    
    def capture_frame(self):
        """捕获一帧图像
        
        Returns:
            np.ndarray or None: 捕获的图像，失败返回None
        """
        if not self.is_opened:
            print(f"摄像头{self.camera_id}未打开")
            return None
        
        try:
            # 捕获图像
            ret, frame = self.cap.read()
            
            if ret:
                return frame
            else:
                print(f"捕获图像失败")
                return None
        except Exception as e:
            print(f"捕获图像失败: {e}")
            return None
    
    def capture_sequence(self, num_frames=10):
        """捕获图像序列
        
        Args:
            num_frames: 捕获的帧数
            
        Returns:
            list: 捕获的图像列表
        """
        frames = []
        
        for i in range(num_frames):
            frame = self.capture_frame()
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def set_resolution(self, resolution):
        """设置摄像头分辨率
        
        Args:
            resolution: 分辨率元组(width, height)
            
        Returns:
            bool: 设置成功返回True，否则返回False
        """
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.resolution = resolution
            print(f"摄像头分辨率已设置为: {resolution}")
            return True
        except Exception as e:
            print(f"设置摄像头分辨率失败: {e}")
            return False
    
    def set_fps(self, fps):
        """设置摄像头帧率
        
        Args:
            fps: 帧率
            
        Returns:
            bool: 设置成功返回True，否则返回False
        """
        try:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.fps = fps
            print(f"摄像头帧率已设置为: {fps}")
            return True
        except Exception as e:
            print(f"设置摄像头帧率失败: {e}")
            return False
    
    def get_status(self):
        """获取摄像头状态
        
        Returns:
            dict: 摄像头状态字典
        """
        return {
            'is_opened': self.is_opened,
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'fps': self.fps,
            'status': 'online' if self.is_opened else 'offline'
        }
    
    def show_preview(self, window_name='Camera Preview'):
        """显示摄像头预览
        
        Args:
            window_name: 预览窗口名称
            
        Returns:
            None
        """
        if not self.is_opened:
            print(f"摄像头{self.camera_id}未打开")
            return
        
        print("按ESC键退出预览")
        
        while True:
            # 捕获图像
            frame = self.capture_frame()
            
            if frame is not None:
                # 显示图像
                cv2.imshow(window_name, frame)
            
            # 等待按键
            key = cv2.waitKey(1)
            
            # 按ESC键退出
            if key == 27:
                break
        
        # 关闭预览窗口
        cv2.destroyWindow(window_name)
    
    def save_frame(self, filename, frame=None):
        """保存图像帧
        
        Args:
            filename: 保存文件名
            frame: 要保存的图像帧，默认为捕获的当前帧
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            if frame is None:
                frame = self.capture_frame()
            
            if frame is not None:
                cv2.imwrite(filename, frame)
                print(f"图像已保存到: {filename}")
                return True
            else:
                print(f"保存图像失败: 图像帧为空")
                return False
        except Exception as e:
            print(f"保存图像失败: {e}")
            return False