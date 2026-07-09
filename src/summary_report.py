def build_summary_report(report, mode, prompt, spatial_report):
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

    if mode == "工地安全检测":
        lines.append("")
        lines.append("=========工地安全风险判断===========")
        lines.append(f"检测到人员: {spatial_report['total_persons']}人")
        lines.append(f"检测到安全帽总数: {spatial_report['detected_helmet_count']}个")
        lines.append(f"判定为已佩戴安全帽: {spatial_report['worn_helmet_count']}人")
        lines.append(f"未与人员匹配的安全帽: {spatial_report['unmatched_helmet_count']}个")
        lines.append(f"检测到反光背心总数: {spatial_report['detected_vest_count']}件")
        lines.append(f"判定为已穿反光背心: {spatial_report['worn_vest_count']}人")
        lines.append(f"未与人员匹配的反光背心: {spatial_report['unmatched_vest_count']}件")
        lines.append("")
        lines.append("风险提示:")

        missing_helmet = spatial_report["missing_helmet_person_count"]
        missing_vest = spatial_report["missing_vest_person_count"]

        if person_count == 0:
            lines.append("未检测到人员，无法进行安全风险判断")
        else:
            if missing_helmet > 0:
                lines.append(f"疑似{missing_helmet}人未佩戴安全帽。")
            else:
                lines.append("所有检测到人员均匹配到安全帽。")

            if missing_vest > 0:
                lines.append(f"疑似{missing_vest}人未穿反光背心。")
            else:
                lines.append("所有检测到人员均匹配到反光背心。")
    
    else:
        lines.append("")
        lines.append("各类别数量:")
        for label, count in report["label_counts"].items():
            lines.append(f"{label}: {count}")

    lines.append("")
    lines.append("说明：当前结果基于 Grounding DINO 检测框和规则判断，仅作为辅助分析，不代表工业级安全检测结果。")

    return "\n".join(lines)