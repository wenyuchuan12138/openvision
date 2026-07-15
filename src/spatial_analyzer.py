# 工地安全风险判断专用逐人空间关系判断

def box_center(box):
    """
    计算bbox中心点。
    bbox格式: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_iou(box1, box2):
    """
    计算两个bbox的IoU。
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

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def is_center_inside_box(center, box):
    """
    判断一个点是否在bbox内。
    """
    cx, cy = center
    x1, y1, x2, y2 = box

    return x1 <= cx <= x2 and y1 <= cy <= y2

def get_head_region(person_box):
    """
    获取人的头部区域。
    简单规则：人的bbox上方35%作为头部区域。
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1

    head_y1 = y1 - height * 0.12
    head_y2 = y1 + height * 0.30

    return [x1, head_y1, x2, head_y2]

def get_body_region(person_box):
    """
    获取人的身体区域。
    简单规则：人的bbox中间30%到85%作为身体区域。
    """
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1

    # 适当放宽身体区域，提升被遮挡背心的匹配可能
    body_x1 = x1 - width * 0.08
    body_x2 = x2 + width * 0.08
    body_y1 = y1 + height * 0.18
    body_y2 = y1 + height * 0.72

    return [body_x1, body_y1, body_x2, body_y2]

def analyze_person_safety_by_spatial_relation(detections):
    """
    基于bbox空间位置关系，逐人判断安全风险。
    判断规则：
    1.helmet的中心点落在person的头部区域内，认为改任佩戴安全帽；
    2.safety vest的中心点落在person的身体区域内，认为该人穿了反光背心。
    """

    persons = [det for det in detections if det["label"] == "person"]
    helmets = [det for det in detections if det["label"] == "helmet"]
    vests = [det for det in detections if det["label"] == "safety vest"]

    used_helmet_indices = set()
    used_vest_indices = set()
    person_results = []

    for indx, person in enumerate(persons, start = 1):
        person_box = person["bbox"]

        head_region = get_head_region(person_box)
        body_region = get_body_region(person_box)

        matched_helmet_index = None
        matched_vest_index = None

        best_helmet_score = 0
        best_vest_score = 0

        # 给当前人员匹配一个尚未使用的安全帽
        for helmet_index, helmet in enumerate(helmets):
            if helmet_index in used_helmet_indices:
                continue

            helmet_box = helmet["bbox"]
            helmet_center = box_center(helmet_box)
            helmet_iou = calculate_iou(head_region, helmet_box)
            helmet_center_inside = 1 if is_center_inside_box(helmet_center, head_region) else 0
            helmet_score = helmet_iou * 0.75 + helmet_center_inside * 0.25

            if helmet_score > best_helmet_score:
                best_helmet_score = helmet_score
                matched_helmet_index = helmet_index

        # 给当前人员匹配一件尚未使用的背心
        for vest_index, vest in enumerate(vests):
            if vest_index in used_vest_indices:
                continue

            vest_box = vest["bbox"]
            vest_center = box_center(vest_box)
            vest_iou = calculate_iou(body_region, vest_box)
            vest_center_inside = 1 if is_center_inside_box(vest_center, body_region) else 0
            vest_score = vest_iou * 0.70 + vest_center_inside * 0.30

            if vest_score > best_vest_score:
                best_vest_score = vest_score
                matched_vest_index = vest_index

        has_helmet = matched_helmet_index is not None and best_helmet_score >= 0.12
        has_vest = matched_vest_index is not None and best_vest_score >= 0.11

        if has_helmet:
            used_helmet_indices.add(matched_helmet_index)
        if has_vest:
            used_vest_indices.add(matched_vest_index)

        risks = []
        
        if not has_helmet:
            risks.append("未检测到已佩戴安全帽")
        
        if not has_vest:
            risks.append("未检测到已穿反光背心")

        if not risks:
            risks.append("未发现明显风险")

        person_results.append({
            "person_id": indx,
            "person_bbox": person_box,
            "has_helmet": has_helmet,
            "has_safety_vest": has_vest,
            "risks": risks
        })

    return {
        "total_persons": len(persons),

        "detected_helmet_count": len(helmets),
        "worn_helmet_count": len(used_helmet_indices),
        "unmatched_helmet_count": len(helmets) - len(used_helmet_indices),
        
        "detected_vest_count": len(vests),
        "worn_vest_count": len(used_vest_indices),
        "unmatched_vest_count": len(vests) - len(used_vest_indices),

        "missing_helmet_person_count": sum(
            not item["has_helmet"] for item in person_results
        ),
        "missing_vest_person_count" : sum(
            not item["has_safety_vest"] for item in person_results
        ),
        "person_results": person_results
    }
