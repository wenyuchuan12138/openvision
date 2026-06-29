from src.postprocess import post_process_detections

def run_prompt_ensemble(
        detector,
        image_path,
        prompt_list,
        threshold = 0.25,
        text_threshold = 0.20,
        ues_construction_postprocess = True
):
    """
    多prompt检查增强

    参考:
    detector: GroundingDINODetector 对象
    image_path: 图片路径
    prompt_list: 多个prompt组成的列表
    threshold: 检测框置信度阈值
    text_threshold: 文本匹配阈值
    ues_construction_postprocess: 是否启用工地场景后处理

    返回：
    final_detections: 融合后的检测结果
    image: 原始图片
    """

    all_detections = []
    image = None
    
    for prompt in prompt_list:
        print(f"正在检测prompt: {prompt}")

        detections, image = detector.predict(
            image_path = image_path,
            text_prompt = prompt,
            threshold = threshold,
            text_threshold = text_threshold
        )

        all_detections.extend(detections)

    if ues_construction_postprocess:
        final_detections = post_process_detections(all_detections)
    else:
        final_detections = all_detections

    return final_detections, image