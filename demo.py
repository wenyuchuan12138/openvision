import os
import requests
from PIL import Image
import matplotlib.pyplot as plt

from src.detector import GroundingDINODetector
from src.visualizer import draw_detections


def download_test_image(save_path):
    """
    下载一张测试图片。
    """

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    response = requests.get(image_url, stream=True)

    image = Image.open(response.raw).convert("RGB")
    image.save(save_path)


def main():
    # 1. 创建文件夹
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 2. 准备测试图片
    image_path = "data/images/test.jpg"

    if not os.path.exists(image_path):
        download_test_image(image_path)

    # 3. 设置文本提示词
    text_prompt = "cat. remote."

    # 4. 加载检测器
    detector = GroundingDINODetector()

    # 5. 执行检测
    detections, image = detector.predict(
        image_path=image_path,
        text_prompt=text_prompt,
        threshold=0.30,
        text_threshold=0.25
    )

    # 6. 打印检测结果
    print("检测结果：")
    for det in detections:
        print(det)

    # 7. 可视化并保存
    result_image = draw_detections(
        image=image,
        detections=detections,
        save_path="outputs/detection_result.jpg"
    )

    # 8. 显示图片
    plt.figure(figsize=(10, 8))
    plt.imshow(result_image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()