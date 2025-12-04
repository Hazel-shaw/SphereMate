import numpy as np
import cv2
import glob

class CameraProjectorCalibrator:
    """相机-投影机标定
    
    用于计算相机和投影机之间的映射关系
    """
    def __init__(self, config=None):
        """初始化标定器
        
        Args:
            config: 标定配置
        """
        self.config = config if config is not None else {}
        
        # 棋盘格参数
        self.checkerboard_size = self.config.get('checkerboard_size', (9, 6))
        self.square_size = self.config.get('square_size', 25.0)  # 单位: mm
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.projector_matrix = None
        self.projector_dist_coeffs = None
        self.camera_projector_matrix = None  # 相机到投影机的变换矩阵
        self.projection_matrix = None  # 最终投影映射矩阵
        
    def find_chessboard_corners(self, images):
        """在图像中寻找棋盘格角点
        
        Args:
            images: 图像列表
            
        Returns:
            tuple: (object_points, image_points)
        """
        # 棋盘格角点的世界坐标
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # 存储所有图像的对象点和图像点
        obj_points = []  # 世界坐标系中的点
        img_points = []  # 图像坐标系中的点
        
        for image in images:
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                # 亚像素级角点优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                obj_points.append(objp)
                img_points.append(corners_refined)
                
                # 绘制角点（可选）
                # cv2.drawChessboardCorners(image, self.checkerboard_size, corners_refined, ret)
                # cv2.imshow('Chessboard', image)
                # cv2.waitKey(500)
        
        # cv2.destroyAllWindows()
        return obj_points, img_points
    
    def calibrate_camera(self, obj_points, img_points, image_shape):
        """标定相机
        
        Args:
            obj_points: 世界坐标系中的点
            img_points: 图像坐标系中的点
            image_shape: 图像形状(H, W)
            
        Returns:
            tuple: (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
        """
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_shape[::-1], None, None
        )
        
        # 计算重投影误差
        mean_error = 0
        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
            mean_error += error
        
        print(f"相机标定完成，重投影误差: {mean_error / len(obj_points):.4f} pixels")
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs
    
    def calibrate_projector(self, obj_points, proj_points, projector_shape):
        """标定投影机
        
        Args:
            obj_points: 世界坐标系中的点
            proj_points: 投影机坐标系中的点
            projector_shape: 投影机分辨率
            
        Returns:
            tuple: (ret, projector_matrix, dist_coeffs, rvecs, tvecs)
        """
        ret, projector_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, proj_points, projector_shape[::-1], None, None
        )
        
        # 计算重投影误差
        mean_error = 0
        for i in range(len(obj_points)):
            proj_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], projector_matrix, dist_coeffs)
            error = cv2.norm(proj_points[i], proj_points2, cv2.NORM_L2) / len(proj_points2)
            mean_error += error
        
        print(f"投影机标定完成，重投影误差: {mean_error / len(obj_points):.4f} pixels")
        
        self.projector_matrix = projector_matrix
        self.projector_dist_coeffs = dist_coeffs
        
        return ret, projector_matrix, dist_coeffs, rvecs, tvecs
    
    def calculate_camera_projector_transform(self, obj_points, camera_points, projector_points):
        """计算相机到投影机的变换矩阵
        
        Args:
            obj_points: 世界坐标系中的点
            camera_points: 相机图像中的点
            projector_points: 投影机图像中的点
            
        Returns:
            numpy.ndarray: 相机到投影机的变换矩阵
        """
        # 使用solvePnP计算相机姿态
        ret, rvec, tvec = cv2.solvePnP(
            obj_points, camera_points, self.camera_matrix, self.dist_coeffs
        )
        
        if not ret:
            print("solvePnP失败")
            return None
        
        # 将旋转向量转换为旋转矩阵
        R, _ = cv2.Rodrigues(rvec)
        
        # 构造相机外参矩阵
        camera_extrinsic = np.hstack((R, tvec))
        
        # 计算投影机姿态
        ret, proj_rvec, proj_tvec = cv2.solvePnP(
            obj_points, projector_points, self.projector_matrix, self.projector_dist_coeffs
        )
        
        if not ret:
            print("投影机solvePnP失败")
            return None
        
        # 将旋转向量转换为旋转矩阵
        proj_R, _ = cv2.Rodrigues(proj_rvec)
        
        # 构造投影机外参矩阵
        proj_extrinsic = np.hstack((proj_R, proj_tvec))
        
        # 计算相机到投影机的变换矩阵
        # 相机到世界：inv(camera_extrinsic)
        # 世界到投影机：proj_extrinsic
        # 相机到投影机：proj_extrinsic * inv(camera_extrinsic)
        camera_extrinsic_hom = np.vstack((camera_extrinsic, [0, 0, 0, 1]))
        proj_extrinsic_hom = np.vstack((proj_extrinsic, [0, 0, 0, 1]))
        
        camera_to_projector = proj_extrinsic_hom @ np.linalg.inv(camera_extrinsic_hom)
        
        self.camera_projector_matrix = camera_to_projector
        
        return camera_to_projector
    
    def compute_projection_matrix(self):
        """计算最终投影映射矩阵
        
        Returns:
            numpy.ndarray: 投影映射矩阵
        """
        if self.camera_matrix is None or self.projector_matrix is None or self.camera_projector_matrix is None:
            print("错误：缺少标定数据")
            return None
        
        # 简化版投影矩阵计算
        # 实际应用中，这里会包含更复杂的映射计算
        self.projection_matrix = np.eye(3)
        
        return self.projection_matrix
    
    def calibrate(self, camera_images, projector_images):
        """完整的标定流程
        
        Args:
            camera_images: 相机拍摄的标定图像列表
            projector_images: 投影机投影的标定图像列表
            
        Returns:
            bool: 标定成功返回True，否则返回False
        """
        try:
            print("开始相机-投影机标定...")
            
            # 1. 标定相机
            print("1. 标定相机...")
            camera_obj_points, camera_img_points = self.find_chessboard_corners(camera_images)
            if len(camera_obj_points) < 10:
                print("错误：相机标定图像不足")
                return False
            
            image_shape = camera_images[0].shape[:2]
            self.calibrate_camera(camera_obj_points, camera_img_points, image_shape)
            
            # 2. 标定投影机
            print("2. 标定投影机...")
            projector_obj_points, projector_img_points = self.find_chessboard_corners(projector_images)
            if len(projector_obj_points) < 10:
                print("错误：投影机标定图像不足")
                return False
            
            projector_shape = self.config.get('projector_resolution', (1920, 1080))
            self.calibrate_projector(projector_obj_points, projector_img_points, projector_shape)
            
            # 3. 计算相机到投影机的变换
            print("3. 计算相机到投影机的变换...")
            # 这里简化处理，实际应用中需要更复杂的匹配
            self.calculate_camera_projector_transform(
                camera_obj_points[0], camera_img_points[0], projector_img_points[0]
            )
            
            # 4. 计算最终投影映射
            print("4. 计算最终投影映射...")
            self.compute_projection_matrix()
            
            print("相机-投影机标定完成！")
            return True
        except Exception as e:
            print(f"标定失败: {e}")
            return False
    
    def save_calibration(self, filename):
        """保存标定结果
        
        Args:
            filename: 保存文件名
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            calibration_data = {
                'checkerboard_size': self.checkerboard_size,
                'square_size': self.square_size,
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'projector_matrix': self.projector_matrix,
                'projector_dist_coeffs': self.projector_dist_coeffs,
                'camera_projector_matrix': self.camera_projector_matrix,
                'projection_matrix': self.projection_matrix
            }
            
            np.savez(filename, **calibration_data)
            print(f"标定结果已保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存标定结果失败: {e}")
            return False
    
    def load_calibration(self, filename):
        """加载标定结果
        
        Args:
            filename: 加载文件名
            
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        try:
            calibration_data = np.load(filename)
            
            self.checkerboard_size = tuple(calibration_data['checkerboard_size'])
            self.square_size = calibration_data['square_size']
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.projector_matrix = calibration_data['projector_matrix']
            self.projector_dist_coeffs = calibration_data['projector_dist_coeffs']
            self.camera_projector_matrix = calibration_data['camera_projector_matrix']
            self.projection_matrix = calibration_data['projection_matrix']
            
            print(f"标定结果已从: {filename} 加载")
            return True
        except Exception as e:
            print(f"加载标定结果失败: {e}")
            return False
    
    def map_point_to_projector(self, camera_point):
        """将相机图像中的点映射到投影机图像
        
        Args:
            camera_point: 相机图像中的点坐标(x, y)
            
        Returns:
            numpy.ndarray: 投影机图像中的对应点坐标
        """
        if self.projection_matrix is None:
            print("错误：投影映射矩阵未计算")
            return None
        
        # 将点转换为齐次坐标
        camera_point_hom = np.array([camera_point[0], camera_point[1], 1.0])
        
        # 应用投影映射
        projector_point_hom = self.projection_matrix @ camera_point_hom
        
        # 转换回非齐次坐标
        projector_point = projector_point_hom[:2] / projector_point_hom[2]
        
        return projector_point