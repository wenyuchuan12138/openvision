# 对检测结果做清洗

# 把不同叫法统一成一个类别
def normalize_label(label):
    """
    同意标签名称。
    """
    # 将标签强制转换为字符串，避免模型返回非字符串类型导致报错
    label = str(label).lower().strip()

    if "person" in label or "worker" in label or "man" in label:
        return "person"
    
    if "helmet" in label or "hard hat" in label or "safety helmet" in label:
        return "helmet"
    
    if "vest" in label or "visibility" in label or "safety" in label or "reflective" in label or "high visibility" in label or "orange vest" in label:
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

    # 获取唯一标签
    # set()存储唯一元素，det()是取出字典中的每一行数据
    labels = set(det["label"] for det in detections)

    for label in labels:
        # 对循环中的一个标签找到其所有结果
        same_label_dets = [det for det in detections if det["label"] == label]
        # 按置信度排序
        same_label_dets = sorted(same_label_dets, key = lambda x: x["score"], reverse = True)
        # sorted(
        # iterable,          # 必需：要排序的列表
        # key=None,         # 可选：排序的键函数，lambda是快速定义简单函数，用什么来做键函数
        # reverse=False     # 可选：是否反向排序，True由大到小
        # )

        kept = []

        while same_label_dets:
            # .pop()移除并返回列表中第一个元素
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

        # 如果无法识别，跳过这个检测
        if new_label is None:
            continue
        
        # .get()获取指定键的值，不存在则返回0.3
        # 获取置信度阈值
        min_score = score_thresholds.get(new_label, 0.30)

        # 过滤低置信度，分数太低就跳过
        if det["score"] < min_score:
            continue
        
        # 保留符合条件的检测
        processed.append({
            "label": new_label,
            "score": det["score"],
            "bbox": det["bbox"]
        })

    # 删除与高置信度框重合的框
    processed = nms_by_label(processed, iou_threshold = 0.5)

    return processed