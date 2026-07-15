def box_center(box):
    """
    计算bbox中心点
    bbox格式:[x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2)/2

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
    uesd_vest_indices = set()

    person_results = []

    for person_id, person in enumerate(persons, start = 1):
        person_box = person["bbox"]

        head_region = get_head_region(person_box)
        body_region = get_body_region(person_box)

        has_helmet = False
        has_vest = False

        best_helmet_ratio = 0
        best_vest_ratio = 0

        matched_helmet_index = None
        matched_vest_index = None

        # 匹配安全帽mask
        for index, helmet in enumerate(helmet_masks):
            if index in used_helmet_indices:
                continue

            ratio = mask_match_ratio(
                mask = helmet["mask"],
                box = head_region
            )

            if ratio > best_helmet_ratio:
                best_helmet_ratio = ratio
                matched_helmet_index = index

        # 匹配反光背心mask
        for index, vest in enumerate(vest_masks):
            if index in uesd_vest_indices:
                continue

            ratio = mask_match_ratio(
                mask = vest["mask"],
                box = body_region
            )

            if ratio > best_vest_ratio:
                best_vest_ratio = ratio
                matched_vest_index = index

        # 阈值可以后续调参
        if matched_helmet_index is not None and best_helmet_ratio >= 0.35:
            has_helmet = True
            used_helmet_indices.add(matched_helmet_index)

        if matched_vest_index is not None and best_vest_ratio >= 0.30:
            has_vest = True
            uesd_vest_indices.add(matched_vest_index)

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
        "matched_vest_mask": len(uesd_vest_indices),
        "unmatched_vest_masks": len(vest_masks) - len(uesd_vest_indices),

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

               