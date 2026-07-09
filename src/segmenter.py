import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor

class SAMSegmenter:
    """
    使用SAM根据Grounding DINO检测框生成分割mask
    """

    def __init__(
            self,
            model_id = "facebook/sam-vit-base",
            device = None
    ):
    
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id

        print(f"正在加载SAM: {model_id}")
        print(f"SAM当前设备: {device}")

        self.processor = SamProcessor.from_pretrained(model_id)

        self.model = SamModel.from_pretrained(
            model_id
        ).to(device)

        self.model.eval()

    def segment(self, image, detections):
        """
        根据检测框对图片进行分割

        参数:
        image:
            PIL.Image 格式的原始图片
        
        detections:
            Grounding DINO 检测结果,例如:
            [
                {
                "label": "person",
                "score": 0.8,
                "bbox": [x1, y1, x2, y2]
                }
            ]
        
        返回:
        segmentation_results:
            每个目标对应的mask、标签、置信度和bbox
        """

        if not detections:
            return []
        
        boxes = [
            detections["bbox"]
            for detection in detections
        ]

        input_boxes = [boxes]

        inputs = self.processor(
            images = image,
            input_boxes = input_boxes,
            return_tensors = "pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_msaks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]

        iou_scores = outputs.iou_scores.cpu()[0]

        segmentation_results = []

        for index, detection in enumerate(detections):
            candidate_masks = masks[index]
            candidate_scores = iou_scores[index]

            best_index = torch.argmax(candidate_scores).item()

            best_mask = candidate_masks[best_index]
            best_score = candidate_scores[best_index].item()

            mask_array = best_mask.numpy().astype(bool)

            segmentation_results.append({
                "label": detections["label"],
                "detection_score": detection["score"],
                "segmentation_score": round(best_score, 4),
                "bbox": detection["bbox"],
                "mask": mask_array
            })

        return segmentation_results