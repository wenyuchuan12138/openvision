import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class GroundingDINODetector:
    """
    Grounding DINO 开放词汇目标检测器。

    功能：
    1. 加载 Hugging Face 版本的 Grounding DINO；
    2. 接收图片和文本 prompt；
    3. 输出检测框、标签和置信度。
    """

    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device=None):
        """
        初始化模型。

        参数：
        model_id：Hugging Face 上的模型名称。
        device：运行设备，默认自动选择 cuda 或 cpu。
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id

        print(f"正在加载模型：{model_id}")
        print(f"当前设备：{self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def predict(self, image_path, text_prompt, threshold=0.30, text_threshold=0.25):
        """
        对单张图片进行开放词汇目标检测。

        参数：
        image_path：图片路径。
        text_prompt：文本提示词，例如 "cat. remote."。
        threshold：检测框置信度阈值。
        text_threshold：文本匹配阈值。

        返回：
        detections：检测结果列表。
        image：原始 PIL 图片。
        """

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )

        result = results[0]

        detections = []

        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            detections.append({
                "label": label,
                "score": round(score.item(), 4),
                "bbox": [round(x, 2) for x in box.tolist()]
            })

        return detections, image