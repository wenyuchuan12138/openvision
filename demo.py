import os
import json
import requests
import urllib3
from PIL import Image
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecurePlatformWarning)

from src.detector import GroundingDINODetector
from src.visualizer import draw_detections
from src.report import gengerate_report
from src.risk_analyzer import analyze_construction_safety

def download_test_image(save_path):
    """
    下载一张测试图片。
    这张图片来自COCO数据集，图中有猫和控制器。
    """

    image_url = "https://images.cocodataset.org/val2017/000000039769.jpg"
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(image_url,
                            headers = headers,
                            timeout = 30,
                            verify = False
                        )
    
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)
    
    # 检测是否真的是图片
    image = Image.open(response.raw).convert("RGB")
    image.save(save_path)

    print("测试图片下载成功：", save_path)

def print_report(report):
    """
    在控制台打印统计报告。
    """

    print("\n=========检测报告===========")
    print(f"图片尺寸:{report['image_width']} x {report['image_height']}")
    print(f"检测目标总数：{report['total_objects']}")

    print("\n各类别数量：")
    for label, count in report ["label_counts"].items():
        print(f"-{label}: {count}")

    print("\n详细目标信息：")
    for idx, obj in enumerate(report["objects"], start = 1):
        print(
            f"{idx}. 类别：{obj['label']},"
            f"置信度: {obj['score']},"
            f"框： {obj['bbox']},"
            f"面积占比: {obj['area_ratio']}"
        )
    
    print("======================================\n")

def main():
    # 创建必要文件夹
    os.makedirs("data/images", exist_ok = True)
    os.makedirs("outputs", exist_ok = True)

    # 准备测试图片
    image_path = "data/images/test.jpg"

    if not os.path.exists(image_path):
        print("正在下载测试图片...")
        download_test_image(image_path)

    # 设置开放词汇检测提示词
    # 注意Grounding DINO的prompt建议用英文，并且每个类别后面加英文句号
    text_prompt = "person. helmet. safety vest."

    print("仓前检测提示词：", text_prompt)

    # 加载Grounding DINO检测器
    detector = GroundingDINODetector()

    # 执行检测
    detections, image = detector.predict(
        image_path = image_path,
        text_prompt = text_prompt,
        threshold = 0.30,
        text_threshold = 0.25
    )

    # 生成检测报告
    report = gengerate_report(
        detections = detections,
        image_size = image.size,
        save_path = "outputs/report.jason"
    )

    risk_report = analyze_construction_safety(report)

    with open("outputs/risk_report.json", "w", encoding = "utf-8") as f:
        json.dump(risk_report, f, ensure_ascii = False, indent = 4)

    # 打印报告
    print_report(report)

    print("\n==========工地安全风险分析=============")
    print(f"人员数量：{risk_report['person_count']}")
    print(f"安全帽数量：{risk_report['helmet_count']}")
    print(f"反光背心数量：{risk_report['vest_count']}")

    print("\n风险提示：")
    for msg in risk_report["risk_messages"]:
        print("-", msg)
    
    print("======================\n")

    # 绘制检测卡并保持结果图片
    result_image = draw_detections(
        image = image,
        detections = detections,
        save_path = "outputs/detection_result.jpg"
    )

    print("检测结果图片已保存到：outputs/detection_result.jpg")
    print("检测统计报告已保存到：outputs/report.json")

    # 在notebook/colab中显示图片
    plt.figure(figsize = (10, 8))
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main() 