import os
import json
import gradio as gr
from PIL import Image

from src.detector import GroundingDINODetector
from src.visualizer import draw_detections
from src.report import gengerate_report
from src.postprocess import post_process_detections

# 全局加载模型，避免每次都惦记按钮重新加载
detecor = GroundingDINODetector()

def openvision_predict(image, text_prompt, threshold, text_threshold):
    """
    Gradio界面调用的预测函数

    输入：
    image:用户上传的图片
    text_prompt:用户输入的检测词
    threshold:检测框置信度阈值
    text_threshold:文本匹配阈值
    
    输出：
    result_image:画好检测框的图片
    report_text:检测报告文本
    """

    os.makedirs("outputs", exist_ok = True)

    image_path = "outputs/input_image.jpg"
    image.save(image_path)

    detections, pil_image = detecor.predict(
        image_path = image_path,
        text_prompt = text_prompt,
        threshold = threshold,
        text_threshold = text_threshold
    )

    detections = post_process_detections(detections)

    report = gengerate_report(
        detections = detections,
        image_size = pil_image.size,
        save_path = "outputs/report.json"
    )

    result_image = draw_detections(
        image = pil_image,
        detections = detections,
        save_path = "outputs/detection_result.jpg"
    )

    report_text = json.dumps(report, ensure_ascii = False, indent = 4)

    return result_image, report_text

demo = gr.Interface(
    fn = openvision_predict,
    inputs = [
        gr.Image(type = "pil", label = "上传图片"),
        gr.Textbox(
            label = "检测提示词",
            value = "person. helmet. hard hat",
            placeholder = "例如：person. heimet. car."
        ),
        gr.Slider(
            minimum = 0.1,
            maximum = 0.9,
            value = 0.25,
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
    ],
    outputs = [
        gr.Image(type = "pil", label = "检测结果图"),
        gr.Textbox(label = "检测报告 JSON")
    ],
    title = "OpenVision 开放词汇目标检测系统",
    description = "基于Grounding DINO的文本提示驱动目标检测系统。山川图片并输入检测词，即可输出检测框、类别、置信度和统计报告。"   
)

if __name__ == "__main__":
    demo.launch()