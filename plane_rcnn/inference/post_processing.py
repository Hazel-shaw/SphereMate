import numpy as np
import cv2
import torch

class PlanePostProcessor:
    """平面检测结果后处理器
    
    用于处理平面检测结果，包括mask优化、平面参数校准等
    """
    def __init__(self):
        """初始化后处理器"""
        pass
    
    def refine_mask(self, mask, method='morphology'):
        """优化平面mask
        
        Args:
            mask: 原始mask，形状为(H, W)
            method: 优化方法，可选'morphology'或'connected_components'
            
        Returns:
            np.ndarray: 优化后的mask
        """
        if method == 'morphology':
            # 形态学操作：先腐蚀再膨胀
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
        elif method == 'connected_components':
            # 连通组件分析，保留最大的连通区域
            num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
            
            if num_labels > 1:
                # 计算每个连通区域的面积
                areas = [np.sum(labels == i) for i in range(1, num_labels)]
                # 找到最大的连通区域
                max_label = np.argmax(areas) + 1
                # 只保留最大的连通区域
                mask = (labels == max_label).astype(np.bool)
        
        return mask
    
    def calculate_plane_area(self, mask, pixel_size=1.0):
        """计算平面面积
        
        Args:
            mask: 平面mask，形状为(H, W)
            pixel_size: 像素大小(单位: mm/px)，默认为1.0
            
        Returns:
            float: 平面面积
        """
        # 计算mask中的像素数量
        pixel_count = np.sum(mask)
        # 转换为实际面积
        area = pixel_count * (pixel_size ** 2)
        return area
    
    def estimate_plane_orientation(self, normal):
        """估计平面方向
        
        Args:
            normal: 平面法向量，形状为(3,)
            
        Returns:
            str: 平面方向描述
        """
        # 归一化法向量
        normal = normal / np.linalg.norm(normal)
        
        # 定义主要方向
        directions = {
            '地面': np.array([0, 1, 0]),  # 向上
            '天花板': np.array([0, -1, 0]),  # 向下
            '墙面': np.array([1, 0, 0]),  # 向前
        }
        
        # 计算与各方向的夹角
        max_similarity = -1
        plane_type = '其他平面'
        
        for direction_name, direction_vec in directions.items():
            similarity = np.dot(normal, direction_vec)
            similarity = abs(similarity)  # 取绝对值，因为方向可能相反
            
            if similarity > max_similarity:
                max_similarity = similarity
                plane_type = direction_name
        
        # 阈值判断
        if max_similarity < 0.7:  # 夹角大于45度
            plane_type = '其他平面'
        
        return plane_type
    
    def calculate_bbox_from_mask(self, mask):
        """从mask计算边界框
        
        Args:
            mask: 平面mask，形状为(H, W)
            
        Returns:
            np.ndarray: 边界框，格式为[x1, y1, x2, y2]
        """
        coords = np.where(mask)
        
        if len(coords[0]) == 0:
            return np.array([0, 0, 0, 0])
        
        x1 = np.min(coords[1])
        y1 = np.min(coords[0])
        x2 = np.max(coords[1])
        y2 = np.max(coords[0])
        
        return np.array([x1, y1, x2, y2])
    
    def mask_to_polygon(self, mask, epsilon=0.01):
        """将mask转换为多边形
        
        Args:
            mask: 平面mask，形状为(H, W)
            epsilon: 多边形近似精度
            
        Returns:
            list: 多边形顶点列表
        """
        # 转换为OpenCV格式
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 取最大的轮廓
        contour = max(contours, key=cv2.contourArea)
        
        # 多边形近似
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        
        # 转换为列表格式
        polygon_list = polygon.reshape(-1, 2).tolist()
        
        return polygon_list
    
    def merge_overlapping_planes(self, planes, iou_threshold=0.5):
        """合并重叠的平面
        
        Args:
            planes: 平面列表，每个平面包含mask和bbox
            iou_threshold: IoU阈值
            
        Returns:
            list: 合并后的平面列表
        """
        if len(planes) <= 1:
            return planes
        
        # 按照面积降序排序
        planes_sorted = sorted(planes, key=lambda p: np.sum(p['mask']), reverse=True)
        
        merged_planes = []
        
        for plane in planes_sorted:
            # 检查是否与已合并的平面重叠
            overlap = False
            for merged_plane in merged_planes:
                # 计算IoU
                iou = self.calculate_iou(plane['mask'], merged_plane['mask'])
                if iou > iou_threshold:
                    overlap = True
                    break
            
            if not overlap:
                merged_planes.append(plane)
        
        return merged_planes
    
    def calculate_iou(self, mask1, mask2):
        """计算两个mask的IoU
        
        Args:
            mask1: 第一个mask，形状为(H, W)
            mask2: 第二个mask，形状为(H, W)
            
        Returns:
            float: IoU值
        """
        # 计算交集
        intersection = np.sum(np.logical_and(mask1, mask2))
        # 计算并集
        union = np.sum(np.logical_or(mask1, mask2))
        
        if union == 0:
            return 0.0
        
        # 计算IoU
        iou = intersection / union
        return iou
    
    def filter_small_planes(self, planes, min_area=100):
        """过滤小面积平面
        
        Args:
            planes: 平面列表
            min_area: 最小面积阈值(单位: 像素)
            
        Returns:
            list: 过滤后的平面列表
        """
        filtered_planes = []
        
        for plane in planes:
            area = np.sum(plane['mask'])
            if area >= min_area:
                filtered_planes.append(plane)
        
        return filtered_planes
    
    def post_process_results(self, results, config=None):
        """完整的结果后处理流程
        
        Args:
            results: 检测结果字典
            config: 后处理配置
            
        Returns:
            dict: 后处理后的检测结果
        """
        if config is None:
            config = {
                'refine_mask': True,
                'mask_refine_method': 'morphology',
                'merge_overlapping': True,
                'iou_threshold': 0.5,
                'filter_small': True,
                'min_area': 100,
                'pixel_size': 1.0
            }
        
        processed_results = results.copy()
        planes = processed_results['planes']
        
        # 1. 优化每个平面的mask
        if config['refine_mask']:
            for plane in planes:
                plane['mask'] = self.refine_mask(plane['mask'], method=config['mask_refine_method'])
                # 重新计算边界框
                plane['bbox'] = self.calculate_bbox_from_mask(plane['mask'])
        
        # 2. 计算平面面积
        for plane in planes:
            plane['area'] = self.calculate_plane_area(plane['mask'], config['pixel_size'])
        
        # 3. 估计平面方向
        for plane in planes:
            plane['orientation'] = self.estimate_plane_orientation(plane['normal'])
        
        # 4. 合并重叠平面
        if config['merge_overlapping']:
            planes = self.merge_overlapping_planes(planes, config['iou_threshold'])
        
        # 5. 过滤小面积平面
        if config['filter_small']:
            planes = self.filter_small_planes(planes, config['min_area'])
        
        # 6. 更新结果
        processed_results['planes'] = planes
        processed_results['post_process_config'] = config
        
        return processed_results