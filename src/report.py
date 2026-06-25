# 把模型输出的检测框，整理成统计报告

import json
from collections import Counter

def gengerate_report(detections, image_size, save_path = None):
    """
    根据检测结果生成统计报告 
    """

    width, height = image_size
    image_area = width * height

    labels = [det["label"] for det in detections]
    label_counts = Counter(labels)

    objects = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        # 这是检测框面积不是精准物体面积
        box_area = box_width * box_height
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
        with open(save_path, "w", encoding = "utf-8") as f:
            json.dump(report, f, ensure_ascii = False, indent = 4)

    return report