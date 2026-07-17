# YOLO检测模块 !pip install ultralytics
from ultralytics import YOLO

# 封装YOLO模型，统一管理
class YOLODetector:
    """
    YOLO工地目标检测器

    使用YOLO检测固定类别目标
    """
    # __int__()加载权重
    def __init__(
            self,
            # YOLO("yolo11n.pt")表示加载YOLO nano模型
            model_path = "yolo11n.pt"
    ):
        """
        初始化YOLO模型

        参数:
        madel_path: YOLO权重路径
        """

        self.model = YOLO(model_path)

    # 输入图片，输出label、score、bbox、source
    def predict(
            self,
            image_path,
            conf = 0.3
    ):
        """
        对图片进行YOLO检测

        输入:
        image_path: 图片路径
        conf: 置信度阈值

        输出：
        detections: 统一格式检测结果
        """

        results = self.model(
            image_path,
            conf = conf
        )

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                confidence = (
                    box.conf[0]
                    .cpu()
                    .item()
                )

                cls_id = (
                    int(
                        box.cls[0]
                        .cpu()
                        .item()
                    )
                )

                label = (
                    self.model.names[
                        cls_id
                    ]
                )

                detections.append({
                    "label": label,
                    "score": round(confidence, 4),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "source": "YOLO"
                })

        return detections