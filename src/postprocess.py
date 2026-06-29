# 对检测结果做清洗

# 把不同叫法统一成一个类别
def normalize_label(label):
    """
    同意标签名称
    """
    label = label.lower().strip()

    if "person" in label:
        return "person"
    
    if "helmet" in label or "hard hat" in label:
        return "helmet"
    
    if "vest" in label or "visibility" in label or "safety" in label:
        return "safety vest"
    
    return None

def calculate_iou(box1, box2):
    """
    计算两个bbox的IoU
    bbox:[x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 +area2 - inter_area

    if union_area == 0:
        return 0
    
    return inter_area/union_area


# NMS去重是同一个目标如果被框好几次，只保留置信度最高的框；iou是两个框的重叠程度，重叠程度不高可保留
def nms_by_label(detections, iou_threshold = 0.5):
    """
    对于同一类别重复框左NMS去重
    """
    final_detections = []

    labels = set(det["label"] for det in detections)

    for label in labels:
        same_label_dets = [det for det in detections if det["label"] == label]
        same_label_dets = sorted(same_label_dets, key = lambda x: x["score"], reverse = True)

        kept = []

        while same_label_dets:
            best = same_label_dets.pop(0)
            kept.append(best)

            same_label_dets = [
                det for det in same_label_dets
                if calculate_iou(best["bbox"], det["bbox"]) < iou_threshold
            ]

        final_detections.extend(kept)

    return final_detections

def post_process_detections(detections):
    """
    对Grounding DINO输出结果进行后处理：
    1.标签归一化
    2.低置信度过滤
    3.NMS去重
    """

    processed = []

    # person分数低于0.35就不要，减少误检
    score_thresholds = {
        "person": 0.35,
        "helmet": 0.30,
        "safety vest": 0.30
    }

    for det in detections:
        new_label = normalize_label(det["label"])

        if new_label is None:
            continue

        min_score = score_thresholds.get(new_label, 0.30)

        if det["score"] < min_score:
            continue

        processed.append({
            "label": new_label,
            "score": det["score"],
            "bbox": det["bbox"]
        })

    processed = nms_by_label(processed, iou_threshold = 0.5)

    return processed