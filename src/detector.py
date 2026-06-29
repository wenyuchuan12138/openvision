# 调用Grounding DINO检测目标和检测

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

    # __init__()初始化模型
    def __init__(self, model_id = "IDEA-Research/grounding-dino-tiny", device = None):
        """
        初始化模型。

        参数：
        model_id：Hugging Face 上的模型名称。
        device：运行设备，默认自动选择 cuda 或 cpu。
        """

        if device is None:
            # 如果有GPU就用cuda，如果没有就用cpu
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id

        print(f"正在加载模型：{model_id}")
        print(f"当前设备：{self.device}")
        
        # 预处理器，负责把图片和文字变成模型能看懂的格式
        self.processor = AutoProcessor.from_pretrained(model_id)
        # 真正的Grounding DINO模型，根据图片和文本提示词，找出图片中和文本相关的目标位置
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    # predict接收图片路径，你想检测的内容，检测框置信度阈值，文本匹配阈值
    def predict(self, image_path, text_prompt, threshold = 0.30, text_threshold = 0.25):
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
        
        # 从image_path读取图片，转换成RGB三通道格式
        image = Image.open(image_path).convert("RGB")
        
        # 处理图片和文本，转成PyTorch张量
        inputs = self.processor(
            images = image,
            text = text_prompt,
            # 返回PyTorch格式的数据
            return_tensors="pt"
         # 把数据放到GPU或CPU上
        ).to(self.device)

        # 模型推理
             # 现在只是推理，不训练模型，不需要计算梯度，避免不需要计算
        with torch.no_grad():
            # 把inputs里面的内容传给模型
            outputs = self.model(**inputs)
        
        # 将模型输出经此函数转换成boxes检测框，scores置信度，lables类别
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # 检测框置信度阈值，控制检测框是否保留，太低会导致检测出很多
            threshold = threshold,
            # 文本匹配阈值，控制区域和文本提示词之间的匹配程度，太低会导致误检太多
            text_threshold = text_threshold,
            # 把模型预测框还原到原图尺寸上
            target_sizes = [image.size[::-1]]
        )

        result = results[0]

        detections = []
        
        # 把目标变成字典
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            detections.append({
                "label": label,
                "score": round(score.item(), 4),
                "bbox": [round(x, 2) for x in box.tolist()]
            })

        return detections, image