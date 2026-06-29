import os
import json
import gradio as gr
from PIL import Image

from src.detector import GroundingDINODetector
from src.visualizer import draw_detections
from src.report import gengerate_report
from src.postprocess import post_process_detections
from src.robust_detector import run_prompt_ensemble
from src.summary_report import build_summary_report

# 全局加载模型，避免每次都惦记按钮重新加载
detector = GroundingDINODetector()

def openvision_predict(image, text_prompt, threshold, text_threshold, mode):
    """
    Gradio界面调用的预测函数

    mode:
    1.通用监测:直接使用用户输入prompt
    2.工地安全监测:使用用户输入prompt,并开启工地后处理
    3.工地鲁棒监测:使用多组prompt ensemble,提高检测稳定性
    """

    # os.makedirs()可创建单个多层目录
    os.makedirs("outputs", exist_ok = True)
    # os.makedirs(
    #     name,                    # 必需：目录路径
    #     mode=0o777,             # 可选：目录权限（默认）
    #     exist_ok=False          # 可选：目录存在时报错，True是不报错
    # )

    image_path = "outputs/input_image.jpg"
    # 保存图片
    image.save(image_path)
    # 保存文本 open("input_image.txt", "w").write("text")
    # 保存JSON json.dump(data, open("input_path.json", "w"))

    if mode == "通用检测":
        detections, pil_image = detector.predict(
            image_path = image_path,
            text_prompt = text_prompt,
            threshold = threshold,
            text_threshold = text_threshold
        )
    
    elif mode == "工地安全检测":
        detections, pil_image = detector.predict(
            iamge_path = image_path,
            text_prompt = text_prompt,
            threshold = threshold,
            text_threshold = text_threshold
        )

        detections = post_process_detections(detections)

    elif mode == "工地鲁棒检测":
        prompt_list = [
            "person. worker. construction woker.",
            "helmet. gard hat. safety helmet.",
            "safety vest. reflective vest. orange vest. high visibility vest."
        ]

        detections, pil_image = run_prompt_ensemble(
            detector = detector,
            image_path = image_path,
            prompt_list = prompt_list,
            threshold = threshold,
            text_threshold = text_threshold,
            ues_construction_postprocess = True
        )

    else:
        raise ValueError(f"位置检测模式:{mode}")

    report = gengerate_report(
        detections = detections,
        image_size = pil_image.size,
        save_path = "outputs/report.json"
    )

    report["mode"] = mode
    report["prompt"] = text_prompt

    if mode == "工地鲁棒检测":
        report["prompt_list"] = [
            "person. worker. construction worker.",
            "helmet. hard hat. safety helmet.",
            "safety vest. reflective vest. orange vest. high visibility vest."
        ]

    result_image = draw_detections(
        image = pil_image,
        detections = detections,
        save_path = "outputs/detection_result.jpg"
    )

    # 把python对象转换成JSON，而json.dump()则是直接保存文件
    # report_text = json.dumps(report, ensure_ascii = False, indent = 4)
    # json.dumps(
    #      obj,        ``            # 必需：要转换的 Python 对象
    #      ensure_ascii=False,     # 可选：是否只用 ASCII 字符，False是保留中文
    #      indent=4,               # 可选：缩进空格数（用于格式化）
    #      sort_keys=False,        # 可选：是否按键排序
    #      default=None            # 可选：处理不可序列化的对象
    # )

    summary_text = build_summary_report(
        report = report,
        mode = mode,
        prompt = text_prompt
    )

    return result_image, summary_text

# gr.Interface()快速创建一个Web界面进行交互
demo = gr.Interface(
    # 要调用的函数,用户点击提交时被调用，参数数量=input数量，返回值数量= outputs数量
    fn = openvision_predict,
    # Gradio输入部分
    inputs = [
        # 这里的输入要和line14一一对应，有顺序
        gr.Image(
            type = "pil", # 返回PIL Image对象
            label = "上传图片"
        ),
        gr.Textbox(
            label = "检测提示词",
            value = "person. helmet. hard hat. reflective vest.", # 默认值
            placeholder = "通用检测:cat. dog. car.;工地检测:person. helmet. hard hat." # 提示文字
        ),
        gr.Slider(
            minimum = 0.1,
            maximum = 0.9,
            value = 0.25, # 默认值
            step = 0.05,
            label = "检测框阈值 threshold"
        ),
        gr.Slider(
            minimum = 0.1,
            maximum = 0.9,
            value = 0.20,
            step = 0.05,
            label = "文本匹配阈值 text_threshold"
        ),
        gr.Dropdown(
            choices = ["通用检测", "工地安全检测", "工地鲁棒检测"],
            value = "通用检测",
            label = "检测模式"
        )
    ],
    # Gradio输出部分
    outputs = [
        gr.Image(type = "pil", label = "检测结果图"),
        gr.Textbox(
        label="检测结果摘要",
        lines=20
    )
    ],
    title = "OpenVision 开放词汇目标检测系统",
    description = "基于Grounding DINO的文本提示驱动目标检测系统。上传图片并输入检测词,即可输出检测框、类别、置信度和统计报告。"   
)

if __name__ == "__main__":
    demo.launch(share = True)
    # 方式1：本地访问
    # demo.launch()
    # 访问：http://localhost:7860
    # 方式2：共享链接（可以分享给他人）
    # demo.launch(share=True)
    # 会生成一个临时的公网链接
    # 方式3：指定端口
    # demo.launch(server_name="0.0.0.0", server_port=8000)