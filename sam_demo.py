import os
import json
from PIL import Image

from src.detector import GroundingDINODetector
from src.segmenter import SAMSegmenter
from src.mask_visualizer import draw_segmentation_masks
from src.mask_report import generate_mask_report

def main():
    os.makedirs("outputs", exist_ok = True)

    image_path = "data/images/test.jpg"

    image = Image.open(image_path).convert("RGB")

    text_prompt = "person. helmet. safety vest."

    detector = GroundingDINODetector()

    detections, image = detector.predict(
        image_path = image_path,
        text_prompt = text_prompt,
        threshold = 0.30,
        text_threshold = 0.20
    )

    print(f"检测到目标数量: {len(detections)}")

    segmenter = SAMSegmenter()

    segmentation_results = segmenter.segment(
        image = image,
        detections = detections
    )

    print(f"完成分割目标数量: {len(segmentation_results)}")

    result_image = draw_segmentation_masks(
        image = image,
        segmentation_results = segmentation_results,
        alpha = 0.45,
        save_path = "outputs/sgementation_result.jpg"
    )

    mask_report = generate_mask_report(
        segmentation_results = segmentation_results,
        image_size = image.size
    )

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

    result_image.show()

    print(
        "分割结果已保存:"
        "outputs/segmentation_result.jpg"
    )

    print(
        "分割报告已保存:"
        "outputs/mask_report.json"
    )

if __name__ == "__main__":
    main()