import cv2
import numpy as np

class PlaneVisualizer:
    """平面可视化与投影
    
    用于可视化检测到的平面并生成投影图像
    """
    def __init__(self, config=None):
        """初始化平面可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config if config is not None else {}
        
        # 可视化参数
        self.colors = {
            'plane': (0, 255, 0),        # 平面区域：绿色
            'bounding_box': (255, 0, 0),  # 边界框：蓝色
            'text': (255, 255, 255),      # 文字：白色
            'normal_vector': (0, 0, 255), # 法向量：红色
            'highlight': (255, 255, 0)    # 高亮区域：黄色
        }
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
    
    def visualize_planes(self, image, results):
        """可视化检测到的平面
        
        Args:
            image: 原始图像，形状为(H, W, 3)
            results: 检测结果字典
            
        Returns:
            np.ndarray: 可视化后的图像
        """
        # 复制图像以避免修改原始图像
        vis_image = image.copy()
        
        # 遍历检测到的平面
        for i, plane in enumerate(results['planes']):
            # 1. 高亮平面区域
            vis_image = self._highlight_plane(vis_image, plane)
            
            # 2. 绘制边界框
            vis_image = self._draw_bounding_box(vis_image, plane)
            
            # 3. 添加平面信息文字
            vis_image = self._add_plane_text(vis_image, plane, i)
            
            # 4. 绘制法向量（可选）
            # vis_image = self._draw_normal_vector(vis_image, plane)
        
        return vis_image
    
    def _highlight_plane(self, image, plane, alpha=0.3):
        """高亮平面区域
        
        Args:
            image: 原始图像
            plane: 平面信息字典
            alpha: 透明度(0-1)
            
        Returns:
            np.ndarray: 高亮后的图像
        """
        # 创建高亮mask
        mask = plane['mask'].astype(np.uint8)
        highlight = np.zeros_like(image)
        highlight[mask > 0] = self.colors['highlight']
        
        # 融合高亮到原始图像
        vis_image = cv2.addWeighted(image, 1 - alpha, highlight, alpha, 0)
        
        return vis_image
    
    def _draw_bounding_box(self, image, plane):
        """绘制平面边界框
        
        Args:
            image: 原始图像
            plane: 平面信息字典
            
        Returns:
            np.ndarray: 绘制边界框后的图像
        """
        bbox = plane['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), self.colors['bounding_box'], self.thickness)
        
        return image
    
    def _add_plane_text(self, image, plane, plane_id):
        """添加平面信息文字
        
        Args:
            image: 原始图像
            plane: 平面信息字典
            plane_id: 平面ID
            
        Returns:
            np.ndarray: 添加文字后的图像
        """
        bbox = plane['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # 计算文字位置
        text_x = x1 + 10
        text_y = y1 - 10
        
        # 平面类型
        if 'orientation' in plane:
            plane_type = plane['orientation']
        else:
            plane_type = '平面'
        
        # 平面置信度
        confidence = plane['confidence']
        
        # 生成文本
        text = f"{plane_type} {plane_id+1}: {confidence:.2f}"
        
        # 绘制文字背景
        (text_width, text_height), _ = cv2.getTextSize(text, self.font, self.font_scale, self.thickness)
        cv2.rectangle(image, (text_x - 5, text_y - text_height - 5), 
                      (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
        
        # 绘制文字
        cv2.putText(image, text, (text_x, text_y), self.font, self.font_scale, 
                    self.colors['text'], self.thickness)
        
        # 添加法向量信息
        normal = plane['normal']
        normal_text = f"N: [{normal[0]:.2f}, {normal[1]:.2f}, {normal[2]:.2f}]"
        normal_text_y = text_y + 20
        cv2.putText(image, normal_text, (text_x, normal_text_y), self.font, 0.4, 
                    self.colors['text'], 1)
        
        return image
    
    def _draw_normal_vector(self, image, plane, length=50):
        """绘制平面法向量
        
        Args:
            image: 原始图像
            plane: 平面信息字典
            length: 法向量长度
            
        Returns:
            np.ndarray: 绘制法向量后的图像
        """
        # 计算平面中心
        mask = plane['mask']
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return image
        
        center_y = int(np.mean(coords[0]))
        center_x = int(np.mean(coords[1]))
        
        # 计算法向量终点
        normal = plane['normal']
        end_x = int(center_x + normal[0] * length)
        end_y = int(center_y + normal[1] * length)
        
        # 绘制法向量
        cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), 
                       self.colors['normal_vector'], self.thickness, tipLength=0.3)
        
        return image
    
    def generate_projection_image(self, plane_results, projection_matrix, projector_resolution):
        """生成投影图像
        
        Args:
            plane_results: 平面检测结果
            projection_matrix: 投影变换矩阵
            projector_resolution: 投影机分辨率
            
        Returns:
            np.ndarray: 投影图像
        """
        # 创建空白投影图像
        proj_image = np.zeros((projector_resolution[1], projector_resolution[0], 3), dtype=np.uint8)
        
        # 遍历检测到的平面
        for i, plane in enumerate(plane_results['planes']):
            # 1. 生成平面高亮区域
            proj_image = self._generate_plane_projection(proj_image, plane, projection_matrix)
            
            # 2. 生成平面信息文字
            proj_image = self._generate_plane_text_projection(proj_image, plane, i, projection_matrix)
        
        return proj_image
    
    def _generate_plane_projection(self, proj_image, plane, projection_matrix):
        """生成平面投影区域
        
        Args:
            proj_image: 投影图像
            plane: 平面信息
            projection_matrix: 投影变换矩阵
            
        Returns:
            np.ndarray: 投影图像
        """
        # 简化实现：在投影图像上绘制边界框
        # 实际应用中，这里需要根据投影矩阵将平面mask映射到投影图像
        
        bbox = plane['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # 模拟投影变换
        # 实际应用中，这里需要使用投影矩阵进行坐标转换
        proj_x1 = int(x1 * 1.5)
        proj_y1 = int(y1 * 1.5)
        proj_x2 = int(x2 * 1.5)
        proj_y2 = int(y2 * 1.5)
        
        # 确保坐标在投影图像范围内
        proj_img_h, proj_img_w = proj_image.shape[:2]
        proj_x1 = max(0, min(proj_img_w - 1, proj_x1))
        proj_y1 = max(0, min(proj_img_h - 1, proj_y1))
        proj_x2 = max(0, min(proj_img_w - 1, proj_x2))
        proj_y2 = max(0, min(proj_img_h - 1, proj_y2))
        
        # 绘制高亮区域
        cv2.rectangle(proj_image, (proj_x1, proj_y1), (proj_x2, proj_y2), 
                     self.colors['highlight'], -1)
        
        # 绘制边界框
        cv2.rectangle(proj_image, (proj_x1, proj_y1), (proj_x2, proj_y2), 
                     self.colors['bounding_box'], self.thickness)
        
        return proj_image
    
    def _generate_plane_text_projection(self, proj_image, plane, plane_id, projection_matrix):
        """生成平面信息文字投影
        
        Args:
            proj_image: 投影图像
            plane: 平面信息
            plane_id: 平面ID
            projection_matrix: 投影变换矩阵
            
        Returns:
            np.ndarray: 投影图像
        """
        # 简化实现：在投影图像上添加文字
        bbox = plane['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # 模拟投影变换
        proj_x1 = int(x1 * 1.5)
        proj_y1 = int(y1 * 1.5)
        
        # 确保坐标在投影图像范围内
        proj_img_h, proj_img_w = proj_image.shape[:2]
        text_x = max(10, min(proj_img_w - 100, proj_x1))
        text_y = max(30, min(proj_img_h - 10, proj_y1))
        
        # 平面类型
        if 'orientation' in plane:
            plane_type = plane['orientation']
        else:
            plane_type = '平面'
        
        # 生成提示信息
        prompt_text = f"检测到{plane_type}"
        
        # 绘制文字
        cv2.putText(proj_image, prompt_text, (text_x, text_y), self.font, self.font_scale, 
                    self.colors['text'], self.thickness)
        
        return proj_image
    
    def generate_highlight_projection(self, plane_results, projection_matrix, projector_resolution):
        """生成高亮投影
        
        Args:
            plane_results: 平面检测结果
            projection_matrix: 投影变换矩阵
            projector_resolution: 投影机分辨率
            
        Returns:
            np.ndarray: 高亮投影图像
        """
        # 创建空白投影图像
        proj_image = np.zeros((projector_resolution[1], projector_resolution[0], 3), dtype=np.uint8)
        
        # 遍历检测到的平面
        for plane in plane_results['planes']:
            # 生成高亮区域
            proj_image = self._generate_highlight_projection(proj_image, plane, projection_matrix)
        
        return proj_image
    
    def _generate_highlight_projection(self, proj_image, plane, projection_matrix):
        """生成高亮区域投影
        
        Args:
            proj_image: 投影图像
            plane: 平面信息
            projection_matrix: 投影变换矩阵
            
        Returns:
            np.ndarray: 投影图像
        """
        # 简化实现：生成高亮区域
        bbox = plane['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        
        # 模拟投影变换
        proj_x1 = int(x1 * 1.5)
        proj_y1 = int(y1 * 1.5)
        proj_x2 = int(x2 * 1.5)
        proj_y2 = int(y2 * 1.5)
        
        # 确保坐标在投影图像范围内
        proj_img_h, proj_img_w = proj_image.shape[:2]
        proj_x1 = max(0, min(proj_img_w - 1, proj_x1))
        proj_y1 = max(0, min(proj_img_h - 1, proj_y1))
        proj_x2 = max(0, min(proj_img_w - 1, proj_x2))
        proj_y2 = max(0, min(proj_img_h - 1, proj_y2))
        
        # 绘制高亮区域
        cv2.rectangle(proj_image, (proj_x1, proj_y1), (proj_x2, proj_y2), 
                     self.colors['highlight'], -1)
        
        return proj_image
    
    def save_visualization(self, image, output_path):
        """保存可视化结果
        
        Args:
            image: 可视化图像
            output_path: 输出路径
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存到: {output_path}")
            return True
        except Exception as e:
            print(f"保存可视化结果失败: {e}")
            return False