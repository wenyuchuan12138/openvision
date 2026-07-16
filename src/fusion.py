# 解决同一个安全帽Grounding DINO检测一次，YOLO检测一次
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个检查框的IoU
    
    IoU: Intersection over Union

    作用: 判断两个框是不是同一目标
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0
    
    return intersection / union

def normalize_label(label):
    """
    统一两个模型的类别名称

    返回同一类别
    """

    label = label.lower()

    if ("helmet" in label or "hard hat" in label):
        return "helmet"
    
    if("vest" in label or "reflective" in label or "safety vest" in label):
        return "safety vest"
    
    if ("person" in label or "worker" in label):
        return "person"
    
    return label

def fuse_detections(
        grounding_results,
        yolo_results,
        iou_threshold = 0.5
):
    """
    融合Grounding DINO和YOLO检测结果

    输入:
    grounding_results: Grounding DINO结果
    yolo_results: YOlO结果

    输出:
    fusion_results: 融合后的检测结果

    融合规则:
    1.类别相同
    2.IoU超过阈值
    认为两个模型检测的是同一目标

    保留score更高的结果
    """

    final_results = []

    used_yolo = set()

    # 先加入Grounding DINO结果
    for gd in grounding_results:

        gd["label"] = normalize_label(
            gd["label"]
        )

        best_match = None
        best_iou = 0

        for index, yd in enumerate(yolo_results):
            if index in used_yolo:
                continue

            yd_label = normalize_label(yd["label"])

            if (gd["label"] != yd_label):
                continue

            iou = calculate_iou(gd["bbox"], yd["bbox"])

            if iou > best_iou:
                best_iou = iou
                best_match = index

        # 找到对应YOLO目标
        if (best_match is not None and best_iou > iou_threshold):
            yolo_det = yolo_results[best_match]

            used_yolo.add(best_match)

            if (yolo_det["score"] > gd["score"]):
                best = yolo_det.copy()
            else:
                best = gd.copy()

            best["scource"] = ("Grounding DINO + YOLO")

            final_results.append(best)

        else:
            gd["source"] = ("Grounding DINO")

            final_results.append(gd)

    for index, yd in enumerate(yolo_results):
        if index not in used_yolo:
            yd["label"] = normalize_label(yd["label"])
            yd["source"] = "YOLO"

            final_results.append(yd)

    return final_results
