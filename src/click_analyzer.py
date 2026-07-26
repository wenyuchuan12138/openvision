def find_clicked_object(x, y, detections):
    """
    根据点击坐标寻找对应检测目标

    参数:
    x:点击位置x坐标
    y:点击位置y坐标
    detections:模型检测结果

    返回:
    点击目标的信息
    """

    for index, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]

        # 判断点击是否在检测框内部
        if (x1 <= x <= x2 and y1 <= y <= y2):
            return{
                "目标编号": index + 1,
                "类别": det["label"],
                "置信度": det["score"],
                "检测来源": det.get(
                    "source",
                    "unknown"
                ),
                "检测框": det["bbox"]
            }

    return {
        "提示": "没有点击到检测目标"
    }