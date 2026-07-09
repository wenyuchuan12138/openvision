# 把模型输出的检测框，整理成统计报告

import json
from collections import Counter

def generate_report(detections, image_size, save_path = None):
    """
    根据检测结果生成统计报告 
    """

    width, height = image_size
    image_area = width * height

    # 输出统计类别及其数量
    labels = [det["label"] for det in detections]
    label_counts = Counter(labels)

    objects = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # 计算框面积
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        # 这是检测框面积不是精准物体面积
        box_area = box_width * box_height
        # 计算面积占比
        area_ratio = box_area / image_area if image_area > 0 else 0

        objects.append({
            "label": det["label"],
            "score": det["score"],
            "bbox": det["bbox"],
            "box_area": round(box_area, 2),
            "area_ratio": round(area_ratio, 4)
        })

    report = {
        "image_width": width,
        "image_height": height,
        "total_objects": len(detections),
        "label_counts": dict(label_counts),
        "objects": objects
    }

    if save_path is not None:
        # 把python字典保存为JSON文件
        with open(save_path, "w", encoding = "utf-8") as f:
                                # ensure_ascii = False保证中文不会变成乱码，indent = 4保证JSON文件格式更好看
            json.dump(report, f, ensure_ascii = False, indent = 4)

    return report