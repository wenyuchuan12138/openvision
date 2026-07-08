# 工地安全风险判断专用逐人空间关系判断

def box_center(box):
    """
    计算bbox中心点。
    bbox格式: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box
    return (x1 + x2)/2, (y1 + y2)/2

def is_center_inside_box(center, box):
    """
    判断一个点是否在bbox内。
    """
    cx, cy = center
    x1, y1, x2, y2 =box

    return x1 <= cx <= x2 and y1 <= cy <= y2

def get_head_region(person_box):
    """
    获取人的头部区域。
    简单规则：人的bbox上方35%作为头部区域。
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1

    head_y2 = y1 + height * 0.35

    return [x1, y1, x2, head_y2]

def get_body_region(person_box):
    """
    获取人的身体区域。
    简单规则：人的bbox中间30%到85%作为身体区域。
    """
    x1, y1, x2, y2 = person_box
    height = y2 - y1

    body_y1 = y1 + height * 0.30
    body_y2 = y1 + height * 0.85

    return [x1, body_y1, x2, body_y2]

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

    person_results = []

    for idx, person in enumerate(persons, start = 1):
        person_box = person["bbox"]

        head_region = get_head_region(person_box)
        body_region = get_body_region(person_box)

        has_helmet = False
        has_vest = False

        for helmet in helmets:
            helmet_center = box_center(helmet["bbox"])
            if is_center_inside_box(helmet_center, head_region):
                has_helmet = True
                break

        for vest in vests:
            vest_center = box_center(vest["bbox"])
            if is_center_inside_box(vest_center, body_region):
                has_vest = True
                break

        risks = []
        
        if not has_helmet:
            risks.append("未检测到安全帽")
        
        if not has_vest:
            risks.append("未检测到反光背心")

        if not risks:
            risks.append("未发现明显风险")

        person_results.append({
            "person_id": idx,
            "person_bbox": person_box,
            "has_helmet": has_helmet,
            "has_safety_vest": has_vest,
            "risks": risks
        })

    spatial_report = {
        "total_persons": len(persons),
        "person_results": person_results
    }

    return spatial_report