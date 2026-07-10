# import os
# import json
# import gradio as gr
# from PIL import Image

# from src.detector import GroundingDINODetector
# from src.visualizer import draw_detections
# from src.report import generate_report
# from src.robust_detector import run_prompt_ensemble
# from src.summary_report import build_summary_report
# from src.spatial_analyzer import analyze_person_safety_by_spatial_relation

# # 全局加载模型，避免每次都惦记按钮重新加载
# detector = GroundingDINODetector()

# def openvision_predict(image, text_prompt, threshold, text_threshold, mode):
#     """
#     Gradio界面调用的预测函数

#     mode:
#     1.通用监测:直接使用用户输入prompt
#     2.工地安全监测:使用用户输入prompt,并开启工地后处理
#     3.工地鲁棒监测:使用多组prompt ensemble,提高检测稳定性
#     """
#     if image is None:
#         raise gr.Error("请先上传图片。")

#     # os.makedirs()可创建单个多层目录
#     os.makedirs("outputs", exist_ok = True)
#     # os.makedirs(
#     #     name,                    # 必需：目录路径
#     #     mode=0o777,             # 可选：目录权限（默认）
#     #     exist_ok=False          # 可选：目录存在时报错，True是不报错
#     # )

#     image_path = "outputs/input_image.jpg"
#     # 保存图片
#     image.save(image_path)
#     # 保存文本 open("input_image.txt", "w").write("text")
#     # 保存JSON json.dump(data, open("input_path.json", "w"))

#     try:
#         if mode == "通用检测":
#             detections, pil_image = detector.predict(
#                 image_path = image_path,
#                 text_prompt = text_prompt,
#                 threshold = threshold,
#                 text_threshold = text_threshold
#             )
    
#         elif mode == "工地安全检测":
#             prompt_list = [
#                 "person. worker. construction worker."
#                 "helmet. hard hat. safety helmet."
#                 "safety vest. reflective vest. high visibility vest."
#             ]
#             detections, pil_image = run_prompt_ensemble(
#                 detector = detector,
#                 image_path = image_path,
#                 prompt_list = prompt_list,
#                 threshold = threshold,
#                 text_threshold = text_threshold,
#                 use_construction_postprocess = True
#             )
#         else:
#             raise ValueError(f"位置检测模式:{mode}")


#         report = generate_report(
#             detections = detections,
#             image_size = pil_image.size,
#             save_path = "outputs/report.json"
#         )

#         report["mode"] = mode
#         report["prompt"] = text_prompt

#         spatial_report = None

#         if mode == "工地安全检测":
#             spatial_report = analyze_person_safety_by_spatial_relation(detections)

#         result_image = draw_detections(
#             image = pil_image,
#             detections = detections,
#             save_path = "outputs/detection_result.jpg"
#         )

#     # 把python对象转换成JSON，而json.dump()则是直接保存文件
#     # report_text = json.dumps(report, ensure_ascii = False, indent = 4)
#     # json.dumps(
#     #      obj,        ``            # 必需：要转换的 Python 对象
#     #      ensure_ascii=False,     # 可选：是否只用 ASCII 字符，False是保留中文
#     #      indent=4,               # 可选：缩进空格数（用于格式化）
#     #      sort_keys=False,        # 可选：是否按键排序
#     #      default=None            # 可选：处理不可序列化的对象
#     # )

#         summary_text = build_summary_report(
#             report = report,
#             spatial_report = spatial_report,
#             mode = mode,
#             prompt = text_prompt
#         )

#         return result_image, summary_text
    
#     except Exception as error:
#         error_message = (
#             "运行失败\n\n"
#             f"错误类型: {type(error).__name__}\n"
#             f"错误信息: {error}"
#         )

#         print(error_message)
#         raise gr.Error(error_message)

# # gr.Interface()快速创建一个Web界面进行交互
# demo = gr.Interface(
#     # 要调用的函数,用户点击提交时被调用，参数数量=input数量，返回值数量= outputs数量
#     fn = openvision_predict,
#     # Gradio输入部分
#     inputs = [
#         # 这里的输入要和line14一一对应，有顺序
#         gr.Image(
#             type = "pil", # 返回PIL Image对象
#             label = "上传图片"
#         ),
#         gr.Textbox(
#             label = "检测提示词",
#             value = "person. helmet. hard hat. reflective vest.", # 默认值
#             placeholder = "通用检测:cat. dog. car.;工地检测:person. helmet. hard hat." # 提示文字
#         ),
#         gr.Slider(
#             minimum = 0.1,
#             maximum = 0.9,
#             value = 0.25, # 默认值
#             step = 0.01,
#             label = "检测框阈值 threshold"
#         ),
#         gr.Slider(
#             minimum = 0.1,
#             maximum = 0.9,
#             value = 0.20,
#             step = 0.01,
#             label = "文本匹配阈值 text_threshold"
#         ),
#         gr.Dropdown(
#             choices = ["通用检测", "工地安全检测"],
#             value = "通用检测",
#             label = "检测模式"
#         )
#     ],
#     # Gradio输出部分
#     outputs = [
#         gr.Image(type = "pil", label = "检测结果图"),
#         gr.Textbox(
#         label="检测结果摘要",
#         lines=20
#     )
#     ],
#     title = "OpenVision 开放词汇目标检测系统",
#     description = "基于Grounding DINO的文本提示驱动目标检测系统。上传图片并输入检测词,即可输出检测框、类别、置信度和统计报告。"   
# )

# if __name__ == "__main__":
#     demo.launch(share = True)
#     # 方式1：本地访问
#     # demo.launch()
#     # 访问：http://localhost:7860
#     # 方式2：共享链接（可以分享给他人）
#     # demo.launch(share=True)
#     # 会生成一个临时的公网链接
#     # 方式3：指定端口
#     # demo.launch(server_name="0.0.0.0", server_port=8000)


import os
import json
import gradio as gr

from src.detector import GroundingDINODetector
from src.segmenter import SAMSegmenter

from src.visualizer import draw_detections
from src.mask_visualizer import draw_segmentation_masks

from src.report import generate_report
from src.mask_report import generate_mask_report
from src.summary_report import build_summary_report

from src.robust_detector import run_prompt_ensemble
from src.spatial_analyzer import analyze_person_safety_by_spatial_relation

# 全局加载Grounding DINO
detector = GroundingDINODetector()

# SAM延迟加载变量
# 先设置未None，表示程序启动时暂时不加载SAM
# 用户第一次选择“检测+SAM分割”后，再加载模型

segmenter = None

def get_segmenter():
    """
    获取SAM分割器

    功能：
    1.第一次调用时加载SAM模型
    2.后续调用时直接复用已加载过的模型
    3.避免每次检测都重新加载SAM

    返回：
        SAMSegmenter对象
    """

    global segmenter

    if segmenter is None:
        print("第一次使用SAM,正在加载分割模型......")

        segmenter = SAMSegmenter(
            model_id = "facebook/sam-vit-base"
        )

    return segmenter

def run_detection(
        image_path,
        text_prompt,
        threshold,
        text_threshold,
        detection_mode
):
    """
    统一执行Grounding DINO检测

    参数:
        image_path:
        检测图片路径
        text_prompt:
        用户输入的文本提示词
        threshold:
        检测框置信度阈值
        text_threshold:
        图像区域与文本之间的匹配阈值
        detection_mode:
        "通用检测"或"工地安全检测"

    返回：
        detections:
        检测结果列表
        pil_image:
        PIL格式原始图片
    """

    # 通用检测模式
    # 直接使用用户输入的prompt
    # 不进行工地类别过滤，适合其他物体的识别

    if detection_mode == "通用检测":
        detections, pil_image  = detector.predict(
            image_path = image_path,
            text_prompt = text_prompt,
            threshold = threshold,
            text_threshold = text_threshold
        )
    
    # 工地安全检测模式
    # 使用多组prompt分别检测人员、安全帽、反光背心
    # 合并结果，在robust_detector内完成工地后处理

    elif detection_mode == "工地安全检测":
        prompt_list = [
            "person. worker. construction worker.",
            "helmet. hard hat. safety helmet.",
            "safety vest. reflective vest. high visibility vest."
        ]

        detections, pil_image = run_prompt_ensemble(
            detector = detector,
            image_path = image_path,
            prompt_list = prompt_list,
            threshold = threshold,
            text_threshold = text_threshold,
            use_construction_postprocess = True
        )

    else:
        raise ValueError(f"未知检测模式: {detection_mode}")
    
    return detections, pil_image

def build_mask_summary(mask_report):
    """
    将SAM分割报告转换为适合前端显示的文字摘要

    参数:
        mask_report:
            generate_mask_reprot() 返回的字典
    
    返回:
        字符串形式的分割摘要
    """

    lines = []

    lines.append("========SAM 分割摘要==========")
    lines.append(
        f"完成分割目标数:"
        f"{mask_report['segmented_object_count']}"
    )
    lines.append("")

    for index, obj in enumerate(
        mask_report["objects"],
        start = 1
    ):
        lines.append(f"目标: {index}")
        lines.append(f"类别: {obj['label']}")
        lines.append(
            f"检测置信度:"
            f"{obj['detection_score']}"
        )
        lines.append(
            f"分割质量分数:"
            f"{obj['segmentation_score']}"
        )
        lines.append(
            f"mask像素面积:"
            f"{obj['mask_area']}"
        )
        lines.append(
            f"mask面积占比: {obj['mask_area_ratio']}"
        )
        lines.append("")
    
    return "\n".join(lines)

def openvision_predict(
        image,
        text_prompt,
        threshold,
        text_threshold,
        detection_mode,
        output_mode
):
    """
    Gradio页面提交后调用的主函数

    整体流程:
    1.保存用户上传的图片
    2.调用Grounding DINO检测目标
    3.生成检测报告
    4.根据output_mode决定是否调用SAM
    5.返回检测图、分割图和文字摘要

    参数:
        image:
        Gradio上传后得到PIL图片
        text_prompt:
        用户输入的检测词
        threshold:
        检测框阈值
        text_threshold:
        文本匹配阈值
        detection_mode:
        通用检测或工地安全检测
        output_mode:
        仅目标检测或检测+SAM分割

    返回:
        detection_image:
        带检测框的图片
        segmentation_image:
        叠加SAM mask的图片
        若没有启用SAM,则返回原检测结果图
        summary_text:
        人类可读的检测和分割摘要
    """

    if image is None:
        raise gr.Error("请先上传图片。")
    
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("请输入检测提示词。")
    
    os.makedirs(
        "outputs",
        exist_ok = True
    )

    image_path = "outputs/input_image.jpg"
    image.save(image_path)

    try:
        # 第一步 Grounding DINO检测
        detections, pil_image = run_detection(
            image_path = image_path,
            text_prompt = text_prompt,
            threshold = threshold,
            text_threshold = text_threshold,
            detection_mode = detection_mode
        ) 

        # 第二步 生成基础检测报告
        report = generate_report(
            detections = detections,
            image_size = pil_image.size,
            save_path = "outputs/report.json"
        )

        report["detection_mode"] = detection_mode
        report["output_mode"] = output_mode
        report["prompt"] = text_prompt

        # 第三步 工地场景逐人空间分析
        spatial_report = None

        if detection_mode == "工地安全检测":
            spatial_report = (
                analyze_person_safety_by_spatial_relation(detections)
            )

        with open(
            "outputs/spatial_report.json",
            "w",
            encoding = "utf-8"
        )as file:
            json.dump(
                spatial_report,
                file,
                ensure_ascii = False,
                indent = 4
            )

        # 第四步 绘制Grounding DINO检测框
        detection_image = draw_detections(
            image = pil_image,
            detections = detections,
            save_path = "outputs/detection_result.jpg"
        )

        # 第五步 生成检测报告
        summary_text = build_summary_report(
            report = report,
            spatial_report = spatial_report,
            mode = detection_mode,
            prompt = text_prompt
        )

        # 第六步 根据输出模式决定是否调用SAM
        if output_mode == "检测+SAM分割":
            if len(detections) == 0:
                summary_text += (
                    "\n\n===========SAM 分割摘要============\n"
                    "Grounding DINO未检测到目标,"
                    "因为没有可提供给SAM的检测框"
                )

                return (
                    detection_image,
                    detection_image,
                    summary_text
                )
            
            # 获取SAM
            # 第一次使用时加载
            sam_segmenter = get_segmenter()

            # 使用Grounding DINO检测框作为SAM提示
            segmentation_results = sam_segmenter.segment(
                image = pil_image,
                detections = detections
            )

            # 将mask叠加到原始图片
            segmentation_image = draw_segmentation_masks(
                image = pil_image,
                segmentation_results = segmentation_results,
                alpha = 0.45,
                save_path = "outputs/segmentation_result.jpg"
            )

            # 根据mask计算像素面积和面积占比
            mask_report = generate_mask_report(
                segmentation_results = segmentation_results,
                image_size = pil_image.size
            )

            # 保存完整mask报告
            with open(
                "outputs/mask_report.json",
                "w",
                encoding = "utf-8"
            ) as file:
                json.dump(
                    mask_report,
                    file,
                    ensure_ascii = False,
                    indent = 4
                )

            # 把SAM摘要追加在检测摘要之后
            mask_summary = build_mask_summary(mask_report)

            summary_text = (
                summary_text
                + "\n\n"
                + mask_summary
            )

        else:
            # 仅检测模式下没有SAM msak图
            # 为了保证Gradio三个输出都有内容
            # 第二张结果图暂时返回检测框图片
            segmentation_image = detection_image

            summary_text += (
                "\n\n当前输出方式: 仅目标检测,"
                "未运行SAM分割"
            )
        
        return (
            detection_image,
            segmentation_image,
            summary_text
        )
    
    except Exception as error:
        error_message = (
            "运行失败\n\n"
            f"错误类型: {type(error).__name__}\n"
            f"错误信息：{error}"
        )

        print(error_message)

        raise gr.Error(error_message)
    
# Gradio 页面
demo = gr.Interface(
    fn = openvision_predict,

    inputs = [
        gr.Image(
            type = "pil",
            label = "上传图片"
        ),

        gr.Textbox(
            label = "检测提示词",
            value = (
                "person. helmet. hard hat."
                "reflective vest."
            ),
            placeholder = (
                "通用检测示例: cat. dog. car."
                "工地检测示例: person. helmet. hard hat."
            )
        ),

        gr.Slider(
            minimum = 0.1,
            maximum = 0.9,
            value = 0.29,
            step = 0.01,
            label = "检测框阈值 threshold"
        ),

        gr.Slider(
            minimum = 0.1,
            maximum = 0.9,
            value = 0.18,
            step = 0.01,
            label = "文本匹配阈值 text_threshold"
        ),

        gr.Dropdown(
            choices = [
                "通用检测",
                "工地安全检测"
            ],
            value  = "通用检测",
            label = "检测模式"
        ),

        gr.Radio(
            choices = [
                "仅目标检测",
                "检测+SAM分割"
            ],
            value = "仅目标检测",
            label = "输出方式"
        )
    ],

    outputs = [
        gr.Image(
            type = "pil",
            label = "Groudning DINO检测结果"
        ),

        gr.Image(
            type = "pil",
            label = "SAM分割结果"
        ),

        gr.Textbox(
            label = "检测与分割摘要",
            lines = 24
        )
    ],

    title = (
        "OpenVision："
        "Grounding DINO + SAM 开放词汇检测分割系统"
    ),

    description = (
        "上传图片并输入检测提示词。"
        "Grounding DINO 负责根据文本定位目标，"
        "SAM 根据检测框生成像素级 mask。"
    )

)

if __name__ == "__main__":
    demo.launch(
        share = True
    )