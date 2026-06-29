def build_summary_report(report, mode, prompt):
    """
    将检测结果转换为适合前端展示的文字摘要
    """

    label_counts = report.get("label_counts", {})
    
    person_count = label_counts.get("person", 0)
    helmet_count = label_counts.get("helmet", 0)
    vest_count = label_counts.get("safety vest", 0)

    lines = []

    lines.append("==========检测摘要============")
    lines.append(f"检测模式:{mode}")
    lines.append(f"检测提示词:{prompt}")
    lines.append(f"图片尺寸:{report.get('image_width')} x {report.get('image_height')}")
    lines.append(f"检测目标总数:{report.get('total_objects')}")
    lines.append("")

    lines.append("各种类别数量:")
    for label, count in label_counts.items():
        lines.append(f"-{label}:{count}")

    if mode in ["工地安全检测", "工地鲁棒检测"]:
        lines.append("")
        lines.append("=========工地安全风险判断===========")
        lines.append(f"人员数量:{person_count}")
        lines.append(f"安全帽数量:{helmet_count}")
        lines.append(f"发光背心数量:{vest_count}")

        lines.append("")
        lines.append("风险提示:")

        if person_count == 0:
            lines.append("未检测到人员，无法进行安全风险判断")
        else:
            if helmet_count < person_count:
                lines.append(f"疑似{person_count - helmet_count}人未佩戴安全帽")
            else:
                lines.append("安全帽数量不少于人员数量，暂时未发现明显安全帽风险")

            if vest_count < person_count:
                lines.append(f"疑似{person_count - vest_count}人未穿反光背心")
            else:
                lines.append("反光数量不少于人员数量，暂时未发现明显反光背心风险")

    lines.append("")
    lines.append("说明：当前结果基于 Grounding DINO 检测框和规则判断，仅作为辅助分析，不代表工业级安全检测结果。")

    return "\n".join(lines)