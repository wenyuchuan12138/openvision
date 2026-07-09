# 调用Grounding DINO检测目标和检测

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# class定义类的关键字，在这里定义了GroundingDINODetector的类，类与函数的区别在于类使用得先定义一个detector = GroundingDINODetector(),才能有result = detector.predict(...)
# 类有完整的系统，包括初始化、属性、多个操作方法
class GroundingDINODetector:
    """
    Grounding DINO 开放词汇目标检测器。

    功能：
    1. 加载 Hugging Face 版本的 Grounding DINO;
    2. 接收图片和文本 prompt;
    3. 输出检测框、标签和置信度。
    """

    # __init__()初始化模型
    def __init__(self, model_id = "IDEA-Research/grounding-dino-base", device = None):
        """
        初始化模型。

        参数：
        model_id:Hugging Face 上的模型名称。
        device:运行设备,默认自动选择 cuda 或 cpu。
        """

        if device is None:
            # torch.cuda.is_available()检查是否有GPU，如果有GPU就用cuda，如果没有就用cpu
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_id = model_id

        print(f"正在加载模型：{model_id}")
        print(f"当前设备：{self.device}")
        
        # 预处理器，AutoProcessor.from_pretrained()负责把图片和文字变成模型能看懂的格式
        # 在这里就可用inputs = self.processor(images = image, text = text_prompt, return_tensors = "pt")
        # 返回值为pixel_values处理后的图片数据 input_ids文本转成的数字编码 attention_mask注意力编码
        # from_pretrained()像从云盘下载一个已训练的AI模型故不需要自己训练，可以直接开始任务
        self.processor = AutoProcessor.from_pretrained(model_id)
        # 真正的Grounding DINO模型，根据图片和文本提示词，找出图片中和文本相关的目标位置
        # AutoModelForZeroShotObjectDetection.from_pretrained(pretrained_model_name_or_path,模型名称或路径cache_dir=None(可选),缓存目录device_map="auto",设备映射(可选))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    # predict接收图片路径，你想检测的内容，检测框置信度阈值，文本匹配阈值
    def predict(self, image_path, text_prompt, threshold = 0.30, text_threshold = 0.25):
        """
        对单张图片进行开放词汇目标检测。

        参数：
        image_path:图片路径。
        text_prompt:文本提示词,例如 "cat. remote."。
        threshold:检测框置信度阈值。
        text_threshold:文本匹配阈值。

        返回：
        detections:检测结果列表。
        image:原始 PIL 图片。
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
            # 把inputs里面的内容传给模型，**自动解包字典，*解包列表/元组
            outputs = self.model(**inputs)
        
        # 将模型输出经此函数转换成boxes检测框，scores置信度，lables类别
        results = self.processor.post_process_grounded_object_detection(
            outputs, # 模型原始输出
            # 将text_prompt的原始文本转换成数字编码
            inputs.input_ids, # 文本编码结果
            # 检测框置信度阈值，控制检测框是否保留，太低会导致检测出很多
            threshold = threshold, # 可选，检测框置信度阈值
            # 文本匹配阈值，控制区域和文本提示词之间的匹配程度，太低会导致误检太多
            text_threshold = text_threshold, # 可选，文本匹配阈值
            # 把模型预测框还原到原图尺寸上，模型格式需要（高度， 宽度），将图片调成这种格式，在这里是反转
            target_sizes = [image.size[::-1]] # 原图尺寸
        )

        result = results[0]

        detections = []
        
        # 把目标变成字典
                               # zip()同时遍历三个列表
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            detections.append({
                "label": label,
                            # score.item()提取张量值，score是张量不是python数字
                "score": round(score.item(), 4),
                        # round()四舍五入，round(x, 2)保留两位小数
                                            # box.tolist()张量转成列表
                "bbox": [round(x, 2) for x in box.tolist()]
                # 具体过程
                # box.tolist() = [50.523, 30.213, 200.347, 150.842]
                # round(50.523, 2) = 50.52
                # round(30.213, 2) = 30.21
                # round(200.347, 2) = 200.35
                # round(150.842, 2) = 150.84
                # 结果 = [50.52, 30.21, 200.35, 150.84]
            })
            # 第1次迭代
            # box = tensor([50.5, 30.2, 200.3, 150.8])
            # score = tensor(0.9523)
            # label = "cat"
    
            # detection = {
            #       "label": "cat",      # 标签
            #       "score": round(0.9523, 4),        # 0.9523
            #       "bbox": [round(x, 2) for x in [50.5, 30.2, 200.3, 150.8]]         # [50.5, 30.2, 200.3, 150.8]
            # }
                                                    
            # {
            #     "label": "cat",
            #     "score": 0.9523,
            #     "bbox": [50.5, 30.2, 200.3, 150.8]
            # }

        return detections, image