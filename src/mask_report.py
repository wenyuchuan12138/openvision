def generate_mask_report(
        segmentation_results,
        image_size
):
    """
    根据SAM mask计算目标真实像素面积占比
    """

    width, height = image_size
    image_area = width * height

    objects = []

    for result in segmentation_results:
        mask_area = int(result["mask"].sum())

        area_ratio = (
            mask_area / image_area
            if image_area > 0
            else 0 
        ) 

        objects.append({
            "label": result["label"],
            "detection_score": result["detection_score"],
            "segmentation_score": result["segmentation_score"],
            "bbox": result["bbox"],
            "mask_area": mask_area,
            "mask_area_ratio": round(area_ratio, 4)
        })

    return {
        "image_width": width,
        "image_height": height,
        "segmented_object_count": len(objects),
        "objects": objects
    }