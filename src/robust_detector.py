from src.postprocess import post_process_detections

# 多个prompt得到多组结果，合并去重
def run_prompt_ensemble(
        detector,
        image_path,
        prompt_list,
        threshold = 0.25,
        text_threshold = 0.20,
        # 决定是否启用工地专用后处理，False直接返回原始检测结果
        use_construction_postprocess = True
):
    """
    多prompt检查增强

    参考:
    detector: GroundingDINODetector 对象
    image_path: 图片路径
    prompt_list: 多个prompt组成的列表
    threshold: 检测框置信度阈值
    text_threshold: 文本匹配阈值
    use_construction_postprocess: 是否启用工地场景后处理

    返回：
    final_detections: 融合后的检测结果
    image: 原始图片
    """
    # 空列表存放所有prompt检测出来的结果
    all_detections = []
    image = None
    
    # 依次取出prompt_list里的每一个prompt
    for prompt in prompt_list:
        print(f"正在检测prompt: {prompt}")

        # 用当前prompt对图片做一次Grounding DINO检测，返回两个值
        detections, image = detector.predict(
            image_path = image_path,
            text_prompt = prompt,
            threshold = threshold,
            text_threshold = text_threshold
        )

        # append会变成嵌套列表[[第一次监测结果]，[],[]],extend会变成扁平列表[第一个结果1，第二个结果，]
        all_detections.extend(detections)

    # 如果是工地安全检测模式，对结果做后处理
    if use_construction_postprocess:
        final_detections = post_process_detections(all_detections)
    else:
        final_detections = all_detections

    return final_detections, image