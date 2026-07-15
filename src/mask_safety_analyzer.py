def box_center(box):
    """
    计算bbox中心点
    bbox格式:[x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def calculate_iou(box1, box2):
    """
    计算两个bbox的IoU
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


def normalized_distance(value, center, max_dist):
    """
    计算一个横向距离归一化得分。
    """
    if max_dist <= 0:
        return 0
    return max(0.0, 1.0 - abs(value - center) / max_dist)


def is_center_inside_box(center, box):
    """
    判断一个点是否在bbox内部
    """
    cx, cy = center
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def get_head_region(person_box):
    """
    根据 person bbox估计头部区域

    当前规则:
    人框顶部向上扩展10%
    人框上方35%作为头部区域
    这样可以覆盖安全帽略高于人头的情况
    """

    x1, y1, x2, y2 = person_box
    height = y2 - y1

    head_y1 = y1 - height * 0.10
    head_y2 = y1 + height * 0.35

    return [x1, head_y1, x2, head_y2]

def get_body_region(person_box):
    """
    根据 person bbox估计身体区域

    当前规则:
    人框中上部到中下部作为身体区域
    横向略微扩展，适应背心框偏移
    """

    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1

    body_x1 = x1 - width * 0.08
    body_x2 = x2 + width * 0.08
    body_y1 = y1 + height * 0.20
    body_y2 = y1 + height * 0.75

    return [body_x1, body_y1, body_x2, body_y2]

def mask_area_in_box(mask, box):
    """
    计算mask有多少像素落在指定bbox区域

    参数:
    mask: bool类型二维数组，True表示目标区域
    box: [x1, y1, x2, y2]

    返回:
    mask在box内的像素数量
    """

    height, width = mask.shape

    x1, y1, x2, y2 = box

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return 0
    
    region = mask[y1 : y2, x1 : x2]

    return int(region.sum())

def mask_match_ratio(mask, box):
    """
    计算mask中有多少比例落在指定区域内
    """

    total_area = int(mask.sum())

    if total_area == 0:
        return 0
    
    inside_area = mask_area_in_box(mask, box)

    return inside_area / total_area

def analyze_safety_by_mask(detections, segmentation_results):
    """
    基于SAM mask进行人员安全装备关联判断

    判断逻辑:
    1.person仍然来自Grounding DINO的bbox
    2.helmet / safety vest使用SAM生成的mask
    3.如果helmet mask大部分落在某个人头区域，则认为该人佩戴安全帽
    4.如果vest mask大部分落在某个人身体区域，则认为该人穿反光背心

    返回:
    mask_safety_report:更细致的穿戴分析结果
    """

    persons = [
        det for det in detections
        if det["label"] == "person"
    ]

    helmet_masks = [
        result for result in segmentation_results
        if result["label"] == "helmet" 
    ]

    vest_masks = [
        result for result in segmentation_results
        if result["label"] == "safety vest"
    ]

    used_helmet_indices = set()
    used_vest_indices = set()

    person_results = []

    for person_id, person in enumerate(persons, start = 1):
        person_box = person["bbox"]

        head_region = get_head_region(person_box)
        body_region = get_body_region(person_box)

        has_helmet = False
        has_vest = False

        best_helmet_score = 0
        best_helmet_ratio = 0
        best_helmet_iou = 0
        best_helmet_center_inside = False
        best_helmet_x_score = 0

        best_vest_score = 0
        best_vest_ratio = 0
        best_vest_iou = 0
        best_vest_center_inside = False
        best_vest_x_score = 0

        matched_helmet_index = None
        matched_vest_index = None

        person_center_x, _ = box_center(person_box)
        person_width = max(1.0, person_box[2] - person_box[0])

        # 匹配安全帽mask
        for index, helmet in enumerate(helmet_masks):
            if index in used_helmet_indices:
                continue

            ratio = mask_match_ratio(
                mask=helmet["mask"],
                box=head_region
            )
            iou = calculate_iou(helmet["bbox"], head_region)
            helmet_center = box_center(helmet["bbox"])
            center_inside = is_center_inside_box(helmet_center, head_region)
            helmet_x_score = normalized_distance(helmet_center[0], person_center_x, person_width * 0.6)
            score = ratio * 0.55 + iou * 0.25 + (1.0 if center_inside else 0.0) * 0.15 + helmet_x_score * 0.05

            if score > best_helmet_score:
                best_helmet_score = score
                best_helmet_ratio = ratio
                best_helmet_iou = iou
                best_helmet_center_inside = center_inside
                best_helmet_x_score = helmet_x_score
                matched_helmet_index = index

        # 匹配反光背心mask
        for index, vest in enumerate(vest_masks):
            if index in used_vest_indices:
                continue

            ratio = mask_match_ratio(
                mask=vest["mask"],
                box=body_region
            )
            iou = calculate_iou(vest["bbox"], body_region)
            vest_center = box_center(vest["bbox"])
            center_inside = is_center_inside_box(vest_center, body_region)
            vest_x_score = normalized_distance(vest_center[0], person_center_x, person_width * 0.8)
            score = ratio * 0.50 + iou * 0.30 + (1.0 if center_inside else 0.0) * 0.15 + vest_x_score * 0.05

            if score > best_vest_score:
                best_vest_score = score
                best_vest_ratio = ratio
                best_vest_iou = iou
                best_vest_center_inside = center_inside
                best_vest_x_score = vest_x_score
                matched_vest_index = index

        # 阈值可以后续调参
        if matched_helmet_index is not None and (
            best_helmet_center_inside
            and (best_helmet_ratio >= 0.22 and best_helmet_iou >= 0.06
            or best_helmet_score >= 0.26)
        ):
            has_helmet = True
            used_helmet_indices.add(matched_helmet_index)

        if matched_vest_index is not None and (
            best_vest_center_inside
            and (best_vest_ratio >= 0.20 and best_vest_iou >= 0.06
            or best_vest_score >= 0.24)
        ):
            has_vest = True
            used_vest_indices.add(matched_vest_index)

        risks = []

        if not has_helmet:
            risks.append("未匹配到已佩戴安全帽")

        if not has_vest:
            risks.append("未匹配到已穿反光背心")

        if not risks:
            risks.append("未发现明显风险")

        person_results.append({
            "person_id": person_id,
            "person_bbox": person_box,
            "has_helmet": has_helmet,
            "has_safety_vest": has_vest,
            "helmet_match_ratio": round(best_helmet_ratio, 4),
            "vest_match_ratio": round(best_vest_ratio, 4),
            "risks": risks
        })
    
    report = {
        "total_persons": len(persons),

        "detected_helmet_masks": len(helmet_masks),
        "matched_helmet_masks": len(used_helmet_indices),
        "unmatched_helmet_masks": len(helmet_masks) - len(used_helmet_indices),

        "detected_vest_masks": len(vest_masks),
        "matched_vest_masks": len(used_vest_indices),
        "unmatched_vest_masks": len(vest_masks) - len(used_vest_indices),

        "missing_helmet_person_count": sum(
            not item["has_helmet"]
            for item in person_results
        ),

        "missing_vest_person_count": sum(
            not item["has_safety_vest"]
            for item in person_results
        ),

        "person_results": person_results
    }

    return report

               