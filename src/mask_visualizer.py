import numpy as np
from PIL import Image

def draw_segmentation_masks(
        image,
        segmentation_results,
        alpha = 0.45,
        save_path = None
):
    """
    把SAM mask叠加到原图

    参数:
    image:
        PIL图片

    segmentation_results:
        SAM输出结果

    alpha:
        mask透明度

    save_payh:
        保存路径

    返回:
        叠加mask后的PIL图片
    """

    image_array = np.array(image).astype(np.float32)

    overlay = image_array.copy()

    mask_colors = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255]
    ]

    for index, result in enumerate(segmentation_results):
        mask = result["mask"]

        color = np.array(
            mask_colors[index % len(mask_colors)],
            dtype = np.float32
        )

        overlay[mask] = (
            overlay[mask] * (1- alpha) + color * alpha
        )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    result_image = Image.fromarray(overlay)

    if save_path is not None:
        result_image.save(save_path)
    
    return result_image