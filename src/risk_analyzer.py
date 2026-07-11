# 工地安全风险判断专用，根据report生成的数量级风险判断

def analyze_construction_safety(report):
    """
    根据检测报告进行工地安全风险分析。

    当前规则：
    1.如果 person 数量 > helmet 数量， 认为可能有人未佩戴安全帽；
    2.如果 person 数量 > safety vest 数量， 认为可能有人未穿反光背心。

    第一版简单规则划分
    """
     # 返回空字典
    label_counts = report.get("label_counts", {})

    person_count = label_counts.get("person", 0)
    helmet_count = label_counts.get("helmet", 0)
    vest_count = label_counts.get("safety vest", 0)

    risk_messages = []

    if person_count == 0:
        risk_messages.append("未检测到人员，暂无人员安全风险判断。")
    else:
        if helmet_count < person_count:
            missing_helmet = person_count - helmet_count
            risk_messages.append(f"疑似存在{missing_helmet}人未佩戴安全帽。")
        else:
            risk_messages.append("安全帽数量不少于人员数量，暂时未发现明显安全帽风险。")

        if vest_count < person_count:
            missing_vest = person_count - vest_count
            risk_messages.append(f"疑似存在{missing_vest}人未穿反光背心。")
        else:
            risk_messages.append("反光背心数量不少于人员数量，暂时未发现明显反光背心风险。")

    risk_report = {
        "person_count":person_count,
        "helmet_count":helmet_count,
        "safety_vest_count":vest_count,
        "risk_messages":risk_messages
    }

    return risk_report