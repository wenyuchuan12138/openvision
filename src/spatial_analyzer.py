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


def normalized_distance(value, center, max_dist):
    """
    归一化水平距离分数。
    """
    if max_dist <= 0:
        return 0
    return max(0.0, 1.0 - abs(value - center) / max_dist)


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

def get_pose_head_region(pose):
    """
    根据YOLO Pose关键点生成头部区域
    
    COCO关键点:
    0 nose
    1 left eye
    2 right eye
    3 left ear
    4 right ear
    """

    if pose is None:
        return None

    head_points = pose[:5]

    xs = [
        p[0]
        for p in head_points
        if p[0] > 0
    ]

    ys = [
        p[1]
        for p in head_points
        if p[1] > 0
    ]

    if len(xs) == 0:
        return None

    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)

    height = y2 - y1

    return [
        x1 - 20,
        y1 - height * 1.5,
        x2 + 20,
        y2 + 10
    ]

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

def analyze_person_safety_by_spatial_relation(detections, pose_results = None):
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

        # # 如果Pose,优先使用关键点
        # if pose_results:
        #     for pose_person in pose_results:

        #         if calculate_iou(
        #             person_box,
        #             pose_person["bbox"]
        #         ) > 0.5:
        #             pose_head = get_pose_head_region(
        #                 pose_person["keypoints"]
        #             )

        #             if pose_head:
        #                 head_region = pose_head

        #             break

        # 原35%规则 + Pose辅助
        original_head_region = get_head_region(person_box)

        head_region = original_head_region

        if pose_results:
            pose_score_regions = []

            for pose_person in pose_results:
                iou = calculate_iou(
                    person_box,
                    pose_person["bbox"]
                )

                if iou > 0.3:
                    pose_head = get_pose_head_region(
                        pose_person["keypoints"]
                    )

                    if pose_head:
                        pose_score_regions.append(pose_head)

            # Pose只有辅助作用
            if len(pose_score_regions) > 0:
                pose_head = pose_score_regions[0]

                # 两个区域取并集
                head_region = [
                    min(
                        original_head_region[0],
                        pose_head[0]
                    ),

                    min(
                        original_head_region[1],
                        pose_head[1]
                    ),

                    max(
                        original_head_region[2],
                        pose_head[2]
                    ),

                    max(
                        original_head_region[3],
                        pose_head[3]
                    )
                ]

        body_region = get_body_region(person_box)

        matched_helmet_index = None
        matched_vest_index = None

        best_helmet_score = 0
        best_vest_score = 0
        best_helmet_center_inside = False
        best_vest_center_inside = False

        # 给当前人员匹配一个尚未使用的安全帽
        person_center_x, _ = box_center(person_box)
        person_width = max(1.0, person_box[2] - person_box[0])

        for helmet_index, helmet in enumerate(helmets):
            if helmet_index in used_helmet_indices:
                continue

            helmet_box = helmet["bbox"]
            helmet_center = box_center(helmet_box)
            helmet_iou = calculate_iou(head_region, helmet_box)
            helmet_center_inside = is_center_inside_box(helmet_center, head_region)
            helmet_x_score = normalized_distance(helmet_center[0], person_center_x, person_width * 0.6)
            helmet_score = helmet_iou * 0.55 + (1.0 if helmet_center_inside else 0.0) * 0.25 + helmet_x_score * 0.20

            if helmet_score > best_helmet_score:
                best_helmet_score = helmet_score
                best_helmet_center_inside = helmet_center_inside
                best_helmet_iou = helmet_iou
                best_helmet_x_score = helmet_x_score
                matched_helmet_index = helmet_index

        # 给当前人员匹配一件尚未使用的背心
        for vest_index, vest in enumerate(vests):
            if vest_index in used_vest_indices:
                continue

            vest_box = vest["bbox"]
            vest_center = box_center(vest_box)
            vest_iou = calculate_iou(body_region, vest_box)
            vest_center_inside = is_center_inside_box(vest_center, body_region)
            vest_x_score = normalized_distance(vest_center[0], person_center_x, person_width * 0.8)
            vest_score = vest_iou * 0.50 + (1.0 if vest_center_inside else 0.0) * 0.30 + vest_x_score * 0.20

            if vest_score > best_vest_score:
                best_vest_score = vest_score
                best_vest_center_inside = vest_center_inside
                best_vest_iou = vest_iou
                best_vest_x_score = vest_x_score
                matched_vest_index = vest_index

        has_helmet = (
            matched_helmet_index is not None
            and best_helmet_center_inside
            and best_helmet_score >= 0.30
            and best_helmet_iou >= 0.08
            and best_helmet_x_score >= 0.30
        )
        has_vest = (
            matched_vest_index is not None
            and best_vest_center_inside
            and best_vest_score >= 0.28
            and best_vest_iou >= 0.06
            and best_vest_x_score >= 0.30
        )

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
