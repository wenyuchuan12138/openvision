from PIL import ImageDraw


def draw_detections(image, detections, save_path=None):
    """
    在图片上绘制检测框。

    参数：
    image：PIL 图片。
    detections：检测结果列表。
    save_path：保存路径。如果为 None，则不保存。

    返回：
    draw_image：画好检测框的图片。
    """

    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    for det in detections:
        label = det["label"]
        score = det["score"]
        x1, y1, x2, y2 = det["bbox"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{label}: {score}", fill="red")

    if save_path is not None:
        draw_image.save(save_path)

    return draw_image