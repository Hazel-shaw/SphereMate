#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单可用版“平面检测”：
- 使用 torchvision 自带的 maskrcnn_resnet50_fpn 预训练模型（COCO）
- 把我们认为“接近平面”的类别的 mask 高亮出来（桌子、地板、床、沙发等）
- 提供：
    - 类 PlaneMaskRCNNDetector：封装检测 + 可视化
    - main()：从单张图片测试
    - （预留）从摄像头读取的接口
"""
import os
import cv2
import torch
import argparse
import numpy as np
from typing import List, Dict, Any

# PlaneRCNN 主模型（你原来就有的）
from plane_rcnn.model.plane_rcnn import PlaneRCNN, LightweightPlaneRCNN

# ✅ 关键：显式导入 PlaneDetector
#   注意：用绝对导入，避免相对路径搞混
from plane_rcnn.inference.plane_detector import PlaneDetector


def parse_args():
    parser = argparse.ArgumentParser(description="PlaneRCNN inference demo")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图像路径（相对或绝对路径都可以）"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="out.jpg",
        help="输出可视化结果的保存路径"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="推理设备"
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.3,
        help="置信度阈值（过滤低置信度平面）"
    )

    return parser


def select_device(device_arg: str) -> torch.device:
    """根据命令行参数选择设备"""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    # auto 或指定设备不可用时的回退策略
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = parse_args()
    args = parser.parse_args()

    # ===== 路径调试信息 =====
    cwd = os.getcwd()
    image_rel = args.image
    image_abs = os.path.abspath(image_rel)

    print("==== 路径调试信息 ====")
    print("当前工作目录 cwd  :", cwd)
    print("命令行传入的 image:", image_rel)
    print("image 绝对路径    :", image_abs)
    print("相对路径是否存在  :", os.path.exists(image_rel))
    print("绝对路径是否存在  :", os.path.exists(image_abs))
    print("======================")

    # 1. 读取图像
    img = cv2.imread(image_rel)
    if img is None:
        img = cv2.imread(image_abs)

    if img is None:
        print(f"❌ 无法读取图像: {image_rel} / {image_abs}")
        return

    print(f"✅ cv2.imread 成功读取图像，shape = {img.shape}")

    # 2. 选择设备
    device = select_device(args.device)
    print("使用设备:", device)

    # 3. 创建 PlaneDetector
    detector = PlaneDetector(
        model_path=None,          # 先用随机权重 / torchvision 权重
        num_classes=2,
        device=str(device),
        use_lightweight=False,    # 先用完整版本
        backbone_type='resnet50'
    )

    # 4. 进行平面检测
    print("开始平面检测...")
    results = detector.detect(img)
    print(f"检测到 {len(results['planes'])} 个平面")

    # 5. 可视化结果
    vis = detector.visualize_detections(img, results)

    # 6. 保存输出
    out_path = args.out
    cv2.imwrite(out_path, vis)
    print(f"✅ 结果已保存到: {out_path}")


if __name__ == "__main__":
    main()



# ===== 1. COCO 类别名表（和 torchvision 的预训练 Mask R-CNN 对应） =====

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# 我们主观认为比较像“平面”的类别（可以自己增减）
PLANE_LIKE_LABELS = {
    "couch",          # 沙发表面
    "bed",            # 床面
    "dining table",   # 桌面
    "chair",          # 椅面
    "tv", "laptop",   # 屏幕平面
    "bench",
    "book"            # 书本封面
    # 你也可以加 "floor" 类似的，但 COCO 里没有单独 floor 类
}


# ===== 2. 检测器类封装 =====

class PlaneMaskRCNNDetector:
    """
    使用 torchvision.maskrcnn_resnet50_fpn 的“平面高亮检测器”。

    - 不需要自己训练权重（使用 COCO 预训练）
    - 通过挑选类别 + mask 面积 + 置信度来近似找“平面”
    """

    def __init__(
        self,
        device: str = "auto",
        score_thresh: float = 0.7,
        min_mask_area_ratio: float = 0.001,
        max_mask_area_ratio: float = 0.25,  # 最大 mask 面积比例，不超过图片的四分之一
        max_detections: int = 5,
        plane_like_labels: List[str] = None,
    ):
        """
        Args:
            device: "auto" / "cpu" / "cuda"
            score_thresh: 置信度阈值
            min_mask_area_ratio: 最小 mask 面积比例（mask 像素数 / 图像总像素），太小的点点不要
            max_mask_area_ratio: 最大 mask 面积比例，不超过图片的四分之一
            max_detections: 最多保留多少个平面
            plane_like_labels: 认为是“平面”的类别名列表
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.score_thresh = score_thresh
        self.min_mask_area_ratio = min_mask_area_ratio
        self.max_mask_area_ratio = max_mask_area_ratio
        self.max_detections = max_detections

        if plane_like_labels is None:
            self.plane_like_labels = PLANE_LIKE_LABELS
        else:
            self.plane_like_labels = set(plane_like_labels)

        # 加载预训练 Mask R-CNN 模型
        print(f"[INFO] 使用设备: {self.device}")
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] 预训练 Mask R-CNN 模型加载完成")

    # ---------- 预处理 ----------

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        把 BGR OpenCV 图像转换为模型需要的张量 (C, H, W), float32, [0, 1]
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image_rgb).float() / 255.0  # HWC
        tensor = tensor.permute(2, 0, 1)  # CHW
        return tensor

    # ---------- 核心检测逻辑 ----------

    def detect_planes(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        在单帧 BGR 图像中检测“平面-like”区域。

        Returns:
            {
              "planes": [
                {
                  "bbox": np.array([x1, y1, x2, y2]),
                  "mask": np.ndarray(bool, (H, W)),
                  "label": str,
                  "score": float,
                  "color": (B, G, R)
                },
                ...
              ],
              "original_shape": (H, W, 3)
            }
        """
        h, w = image_bgr.shape[:2]
        total_pixels = float(h * w)

        inp = self.preprocess(image_bgr).to(self.device)

        with torch.no_grad():
            preds = self.model([inp])[0]

        boxes = preds["boxes"]       # (N, 4)
        labels = preds["labels"]     # (N,)
        scores = preds["scores"]     # (N,)
        masks = preds["masks"]       # (N, 1, H, W)

        # 先按 score 过滤
        keep = scores >= self.score_thresh
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks = masks[keep]

        planes = []
        for box, lab, score, mask in zip(boxes, labels, scores, masks):
            class_id = int(lab.item())
            if class_id < 0 or class_id >= len(COCO_INSTANCE_CATEGORY_NAMES):
                continue

            class_name = COCO_INSTANCE_CATEGORY_NAMES[class_id]

            # 只保留我们认为“接近平面”的类别
            if class_name not in self.plane_like_labels:
                continue

            # mask: (1, H, W) -> (H, W)
            mask_bin = (mask[0] > 0.5).to("cpu").numpy().astype(np.uint8)
            mask_area = mask_bin.sum()

            # 检查面积是否在有效范围内
            area_ratio = mask_area / total_pixels
            if area_ratio < self.min_mask_area_ratio:
                # 面积太小，跳过
                continue
            if area_ratio > self.max_mask_area_ratio:
                # 面积太大（超过四分之一），跳过
                continue

            x1, y1, x2, y2 = box.to("cpu").numpy()
            bbox = np.array([x1, y1, x2, y2]).astype(int)

            # 随机颜色，方便区分不同平面
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )

            planes.append(
                {
                    "bbox": bbox,
                    "mask": mask_bin.astype(bool),
                    "label": class_name,
                    "score": float(score.item()),
                    "color": color,
                }
            )

        # 按 score 排序，取前 max_detections 个
        planes.sort(key=lambda p: p["score"], reverse=True)
        planes = planes[: self.max_detections]

        return {
            "planes": planes,
            "original_shape": image_bgr.shape,
        }

    # ---------- 可视化 ----------

    def visualize(self, image_bgr: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        用 bbox + 半透明 mask + 文本，把检测结果画在图上。
        """
        vis = image_bgr.copy()

        for i, pl in enumerate(results["planes"]):
            bbox = pl["bbox"]
            mask = pl["mask"]
            label = pl["label"]
            score = pl["score"]
            color = pl["color"]

            x1, y1, x2, y2 = bbox

            # 1) 画 bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 2) 画彩色 mask（半透明）
            colored = np.zeros_like(vis, dtype=np.uint8)
            colored[mask] = color
            vis = cv2.addWeighted(vis, 1.0, colored, 0.4, 0)

            # 3) 写标签文本
            text = f"{label} {score:.2f}"
            cv2.putText(
                vis,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return vis


# ===== 3. 命令行 & main 函数 =====

def parse_args():
    parser = argparse.ArgumentParser(description="Plane-like detection using Mask R-CNN")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="输入图片路径（BGR，OpenCV 可读）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="plane_detection_result.jpg",
        help="输出可视化结果路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="运行设备",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.7,
        help="检测得分阈值（默认 0.7）",
    )
    return parser.parse_args()


import argparse
import os

def main():
    """命令行用法 + 路径调试"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--out", required=True, help="输出结果图像路径")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--score_thresh", type=float, default=0.5)
    args = parser.parse_args()

    # 设备选择
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("==== 路径调试信息 ====")
    print("当前工作目录 cwd  :", os.getcwd())
    print("命令行传入的 image:", args.image)
    abs_image_path = os.path.abspath(args.image)
    print("image 绝对路径    :", abs_image_path)
    print("相对路径是否存在  :", os.path.exists(args.image))
    print("绝对路径是否存在  :", os.path.exists(abs_image_path))
    print("======================")

    # 先检查路径是否存在
    if not os.path.exists(abs_image_path):
        print(f"❌ 文件不存在: {abs_image_path}")
        return

    # 尝试用 cv2 读图
    image = cv2.imread(abs_image_path)
    if image is None:
        print("⚠️ cv2.imread 读取失败，尝试使用 PIL 读取并转换为 BGR")
        try:
            pil_img = Image.open(abs_image_path).convert("RGB")
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            print("✅ 使用 PIL 成功读取图像")
        except Exception as e:
            print(f"❌ PIL 也无法读取图像: {e}")
            return
    else:
        print("✅ cv2.imread 成功读取图像，shape =", image.shape)

    # 创建平面检测器
    detector = PlaneDetector(
        device=device,
        use_lightweight=False
    )

    # 平面检测
    print("开始平面检测...")
    results = detector.detect(image)
    print(f"检测到 {len(results['planes'])} 个平面")

    # 可视化并保存
    vis_image = detector.visualize_detections(image, results)
    out_abs = os.path.abspath(args.out)
    cv2.imwrite(out_abs, vis_image)
    print(f"✅ 可视化结果已保存到: {out_abs}")

if __name__ == "__main__":
    main()
